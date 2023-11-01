# Adapted from vdm/experiment.py. Simplified boilerplate for jax experiments.
# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import functools

from absl import logging
from clu import parameter_overview
from clu import checkpoint
import flax.jax_utils as flax_utils
import flax
from jax._src.random import PRNGKey
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from optax._src import base
import tensorflow as tf
import os
import shutil
import pprint
from clu import metric_writers, periodic_actions
from common.custom_writers import create_default_writer
import inspect
import itertools

import common.utils
import common.data_lib
from common.custom_metrics import Metrics
import plot_utils
import matplotlib.pyplot as plt


class BaseExperiment(ABC):
  """Boilerplate for training and evaluating flax models. Keeps track of config and data etc."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config

    # Set seed before initializing model.
    seed = config.train_eval_config.seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    self.rng = common.utils.with_verbosity("ERROR", lambda: jax.random.PRNGKey(seed))
    if config.train_data_config.get('fixed_batch') or config.eval_data_config.get('fixed_batch'):
      tf.config.experimental.enable_op_determinism()  # Ensure reproducible tf.data.

    # initialize dataset
    logging.info('=== Initializing dataset ===')
    # self.rng, data_rng = jax.random.split(self.rng)
    train_ds = common.data_lib.get_dataset(**config.train_data_config).prefetch(tf.data.AUTOTUNE)
    self.train_iter = iter(train_ds)
    if config.train_data_config.get('fixed_batch'):
      batch = jax.tree_map(jnp.asarray, next(self.train_iter))
      self.train_iter = itertools.repeat(batch)

    if config.eval_data_config.get('reuse_train'):  # Useful when the true source is discrete.
      self.eval_iter = self.train_iter
    else:
      self.eval_ds = common.data_lib.get_dataset(**config.eval_data_config)
      self.eval_iter = iter(self.eval_ds)
      if config.eval_data_config.get('fixed_batch'):
        batch = jax.tree_map(jnp.asarray, next(self.eval_iter))
        self.eval_iter = itertools.repeat(batch)

    # Create/initialize model
    logging.info('=== Creating train state ===')
    self.rng, model_rng = jax.random.split(self.rng)
    self.state = self.init_state(model_rng)
    parameter_overview.log_parameter_overview(self.state.params)

    # Restore from checkpoint
    ckpt_restore_dir = self.config.get('ckpt_restore_dir', 'None')
    if ckpt_restore_dir != 'None':
      ckpt_restore = checkpoint.Checkpoint(ckpt_restore_dir)
      checkpoint_to_restore = ckpt_restore.get_latest_checkpoint_to_restore_from()
      assert checkpoint_to_restore
      state_restore_dict = ckpt_restore.restore_dict(checkpoint_to_restore)
      self.state = restore_partial(self.state, state_restore_dict)
      logging.info("Restored from %s", checkpoint_to_restore)
      del state_restore_dict, ckpt_restore, checkpoint_to_restore

    # initialize train/eval step
    logging.info('=== Initializing train/eval step ===')
    self.rng, train_rng = jax.random.split(self.rng)
    self.p_train_step = functools.partial(self.train_step, train_rng)

    self.rng, eval_rng = jax.random.split(self.rng)
    self.p_eval_step = functools.partial(self.eval_step, eval_rng)

    logging.info('=== Done with Experiment.__init__ ===')

  def get_lr_schedule(self):
    learning_rate = self.config.optimizer_config.learning_rate
    train_eval_config = self.config.train_eval_config
    # Create learning rate schedule
    # Not doing warmup.
    # warmup_fn = optax.linear_schedule(
    #   init_value=0.0,
    #   end_value=learning_rate,
    #   transition_steps=train_eval_config.num_steps_lr_warmup
    # )
    main_lr_fn = optax.constant_schedule(learning_rate)
    if self.config.optimizer_config.lr_decay:
      opt_config = self.config.optimizer_config
      decay_factor = opt_config.decay_factor  # 0.1 means a 10x reduction
      decay_steps = train_eval_config.lr_decay_last_steps_ratio * train_eval_config.num_train_steps
      if opt_config.decay_type == 'const':
        decay_fn = optax.constant_schedule(learning_rate * decay_factor)
      elif opt_config.decay_type == 'linear':
        decay_fn = optax.linear_schedule(init_value=learning_rate, end_value=learning_rate * decay_factor,
                                         transition_steps=decay_steps)
      elif opt_config.decay_type == 'inv_sqrt':
        # lr_t = lr_0 / \sqrt(decay_rate * t + 1)
        decay_rate = (decay_factor ** (-2) - 1) / decay_steps
        decay_fn = lambda t: learning_rate / jnp.sqrt(t * decay_rate + 1)
      else:
        raise NotImplementedError(opt_config.decay_type)
      schedule_fn = optax.join_schedules(
        schedules=[main_lr_fn, decay_fn], boundaries=[
          train_eval_config.num_train_steps - decay_steps]
      )
    else:
      schedule_fn = main_lr_fn

    return schedule_fn

  def get_optimizer(self, lr: float) -> base.GradientTransformation:
    """Get an optax optimizer. Can be overided. """
    config = self.config.optimizer_config

    optimizer = getattr(optax, config.name)(
      learning_rate=lr,
      **config.args
    )  # like "optax.adam(learning_rate=lr, ...)"

    if hasattr(config, "gradient_clip_norm"):
      clip = optax.clip_by_global_norm(config.gradient_clip_norm)
      optimizer = optax.chain(clip, optimizer)

    return optimizer

  @abstractmethod
  def init_state(self, rng: PRNGKey):
    """Return the initial state of the model (and optimizer, if being used)."""
    ...

  def simple_train_eval_loop(self, train_eval_config, workdir):
    # logging.info("TF physical devices:\n%s", str(tf.config.list_physical_devices()))
    logging.info("JAX physical devices:\n%s", str(jax.devices()))
    # config = self.config.train_eval_config
    config = train_eval_config

    # Create writers for logs.
    train_dir = os.path.join(workdir, TRAIN_COLLECTION)
    # train_writer = metric_writers.create_default_writer(train_dir, collection=TRAIN_COLLECTION)
    train_writer = create_default_writer(train_dir)
    train_writer.write_hparams(config.to_dict())

    val_dir = os.path.join(workdir, VAL_COLLECTION)
    val_writer = create_default_writer(val_dir)

    # Get train state. Should only refer to `state` from now on, `self.state` no longer used.
    state = self.state

    # Set up checkpointing of the model and the input pipeline.
    # checkpoint_dir = os.path.join(workdir, 'checkpoints')
    checkpoint_dir = os.path.join(train_dir, CHECKPOINTS_DIR_NAME)
    # ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=5)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    logging.info("Will save checkpoints to %s", checkpoint_dir)
    max_ckpts_to_keep = config.get("max_ckpts_to_keep", 1)
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=max_ckpts_to_keep)
    checkpoint_to_restore = ckpt.get_latest_checkpoint_to_restore_from()
    if checkpoint_to_restore:
      state = ckpt.restore_or_initialize(state, checkpoint_to_restore)
    initial_step = int(state.step)

    # # Distribute training.
    # state = flax_utils.replicate(state)

    hooks = []
    report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=train_writer, every_secs=60)
    if jax.process_index() == 0:
      hooks += [report_progress]
      if config.get('profile'):
        hooks += [periodic_actions.Profile(num_profile_steps=5,
                                           logdir=train_dir)]

    step = initial_step
    substeps = config.substeps  # The number of gradient updates per p_train_step call.

    with metric_writers.ensure_flushes(train_writer):
      logging.info('=== Start training ===')
      # the step count starts from 1 to num_train_steps
      while step < config.num_train_steps:
        is_last_step = step + substeps >= config.num_train_steps
        # One training step
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
          batch = jax.tree_map(jnp.asarray, next(self.train_iter))
          eval_train = (config.eval_every_steps > 0 and (
              step + substeps) % config.eval_every_steps == 0) or is_last_step
          state, train_metrics = self.p_train_step(state, batch, eval_train)

        # Quick indication that training is happening.
        logging.log_first_n(
          logging.WARNING, 'Ran training step %d.', 3, step)
        for h in hooks:
          h(step)

        # new_step = int(state.step[0])
        new_step = int(state.step)  # Above crashes since state.step is a scalar in my unparallel code.
        assert new_step == step + substeps
        step = new_step
        # By now, `step` \in [0, num_train_steps] is the number of gradient steps already taken.

        if config.get('log_imgs_every_steps') and (step % config.log_imgs_every_steps == 0 or is_last_step):
          data_spec = self.config.train_data_config.data_spec
          image_src = data_spec in common.data_lib.IMG_SOURCE_NAMES or np.any(
            [name in data_spec for name in common.data_lib.IMG_SOURCE_NAMES]) or len(jnp.shape(batch)) > 2
          if image_src:
            nu_x_to_plot = state.params['nu_x'][:8]
            figsize = (12, 8)
            cmap = None
            vmin = vmax = None
            if 'mnist' in data_spec:
              cmap = 'gray'
              vmin, vmax = 0, 255
            fig = plot_utils.plot_float_imgs(nu_x_to_plot, figsize=figsize, cmap=cmap, vmin=vmin, vmax=vmax)
            title = f'step={step}'
            plt.gca().set_title(title)
            img = plot_utils.fig_to_np_arr(fig)
            plt.close(fig)
            image_metrics = {'nu_x': img}
          else:
            num_pts_to_plot = 1000
            scatter_img = self.make_scatter_img(state.params['nu_x'][:num_pts_to_plot],
                                                state.params['nu_w'][:num_pts_to_plot],
                                                mu_x=batch[:num_pts_to_plot],
                                                title=f'step={step}')
            image_metrics = {'scatter': scatter_img}
          train_writer.write_images(step, image_metrics)

        if step % config.log_metrics_every_steps == 0 or is_last_step:
          # metrics = flax_utils.unreplicate(_train_metrics['scalars'])
          train_metrics = Metrics(train_metrics['scalars'], train_metrics.get('images', {}))

          # def avg_over_substeps(x):
          #   assert x.shape[0] == substeps
          #   return float(x.mean(axis=0))
          #
          # metrics = jax.tree_map(avg_over_substeps, metrics)
          # writer.write_scalars(step, metrics)
          # assert len(metrics) == substeps
          # print(scalar_metrics)
          # train_writer.write_scalars(step, scalar_metrics[-1])  # This is for when training with multiple substeps.
          train_writer.write_scalars(step, train_metrics.scalars_float)

        if (config.eval_every_steps > 0 and step % config.eval_every_steps == 0) or is_last_step \
            or step == config.log_metrics_every_steps:
          logging.info("Evaluating at step %d", step)
          with report_progress.timed('eval'):
            metrics_list = []
            if config.num_eval_steps is None:  # Then we will iterate through the full eval_ds (assumed finite)
              eval_iter = iter(self.eval_ds)
            else:
              eval_iter = self.eval_iter
            eval_steps_done = 0
            val_size = 0
            for eval_step, batch in enumerate(eval_iter):
              if eval_steps_done == config.num_eval_steps:
                break
              batch = jax.tree_map(jnp.asarray, batch)
              metrics = self.p_eval_step(state.params, batch, eval_step=eval_step)
              metrics = Metrics(metrics['scalars'], metrics.get('images', {}))
              metrics_list.append(metrics)
              eval_steps_done += 1
              val_size += len(batch)
            metrics_list = jax.tree_map(float, metrics_list)  # Convert jnp.Array type to scalars, to make tf happy.
            eval_metrics = Metrics.merge_metrics(metrics_list)
            val_writer.write_scalars(step, eval_metrics.scalars_numpy)
            # val_writer.write_images(step, eval_metrics.images)
            logging.info("Ran validation on %d instances.", val_size)

        if step % config.checkpoint_every_steps == 0 or is_last_step:
          with report_progress.timed('checkpoint'):
            # ckpt.save(flax_utils.unreplicate(state))
            ckpt.save(state)

      logging.info('=== Finished training ===')

    train_writer.close()  # Will make gif.
    val_writer.close()  # Will make gif.

    return None

  def train_and_evaluate(self, experiments_dir: str, runname: str):
    ##################### BEGIN: slurm-based workdir setup and good old bookkeeping #########################
    xid = common.utils.get_xid()
    # Here, each runname is associated with a different work unit (Slurm call this a 'array job task')
    # within the same experiment. We add the work unit id prefix to make it easier to warm start
    # with the matching wid later.
    wid = common.utils.get_wid()
    if wid is None:
      wid_prefix = ''
    else:
      wid_prefix = f'wid={wid}-'
    workdir = os.path.join(experiments_dir, xid, wid_prefix + runname)
    if not os.path.exists(workdir):
      os.makedirs(workdir)
    # absl logs from this point on will be saved to files in workdir.
    logging.get_absl_handler().use_absl_log_file(program_name="trainer", log_dir=workdir)

    logging.warning('=== Experiment.train_and_evaluate() ===')
    config = self.config
    logging.info("Using workdir:\n%s", workdir)
    logging.info("Input config:\n%s", pprint.pformat(config))

    if config.train_data_config.get('fixed_batch'):
      logging.info('Fixed train batch mean: %f', next(self.train_iter).mean().item())
    if config.eval_data_config.get('fixed_batch'):
      logging.info('Fixed eval batch mean: %f', next(self.eval_iter).mean().item())

    # Save the config provided.
    with open(os.path.join(workdir, f"config.json"), "w") as f:
      f.write(config.to_json(indent=2))
    if "config_filename" in config:
      shutil.copy2(config["config_filename"], os.path.join(experiments_dir, xid, "config_script.py"))

    # Log more info.
    logging.info("Run info:\n%s", pprint.pformat(common.utils.get_run_info()))
    common.utils.log_run_info(workdir=workdir)
    # Write a copy of models source code.
    model_source_str = inspect.getsource(inspect.getmodule(self))
    with open(os.path.join(workdir, f"main.py"), "w") as f:
      f.write(model_source_str)
    ##################### BEGIN: slurm-based workdir setup and good old bookkeeping #########################

    return self.simple_train_eval_loop(self.config.train_eval_config, workdir)

  def evaluate(self, logdir, checkpoint_dir):
    ...

  def train_step(self, base_rng, state, batch, eval=False):
    # return new_state, metrics
    ...

  def eval_step(self, base_rng, params, batch, eval_step=0):
    # return metrics
    ...

  def make_scatter_img(self, nu_x, nu_w=None, mu_x=None, coords_to_plot=(0, 1), title=None):
    """
    Plot data and the \nu particles.
    :param nu_x:
    :param nu_w:
    :param mu_x:
    :param coords_to_plot:
    :param title:
    :return:
    """
    assert len(jnp.shape(mu_x)) == 2, 'Expected flat data matrix, got one with shape ' + str(mu_x.shape)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    i, j = coords_to_plot
    if mu_x is not None:
      ax.scatter(mu_x[:, i], mu_x[:, j], marker='.', alpha=0.3, label=r'$\mu$')

    if nu_w is not None and jnp.std(nu_w) > 1e-5:
      # Make sure the color isn't too faint, since the nu_w can be very close to 0.
      min_w_to_plot = 0.1
      nu_w_c = (1 - min_w_to_plot) * nu_w + min_w_to_plot
      ax.scatter(nu_x[:, i], nu_x[:, j], c=nu_w_c, cmap='Oranges', vmin=0, marker='x', label=r'$\nu$')
    else:
      ax.scatter(nu_x[:, i], nu_x[:, j], marker='x', label=r'$\nu$')

    ax.legend()
    if not hasattr(self, '_scatter_xlim'):
      self._scatter_xlim = ax.get_xlim()
    if not hasattr(self, '_scatter_ylim'):
      self._scatter_ylim = ax.get_ylim()
    ax.set_xlim(self._scatter_xlim)
    ax.set_ylim(self._scatter_ylim)
    ax.set_aspect('equal')

    if title:
      ax.set_title(title)

    img = plot_utils.fig_to_np_arr(fig)
    plt.close(fig)

    return img


def copy_dict(dict1, dict2):
  if not isinstance(dict1, dict):
    assert not isinstance(dict2, dict)
    return dict2
  for key in dict1.keys():
    if key in dict2:
      dict1[key] = copy_dict(dict1[key], dict2[key])

  return dict1


def restore_partial(state, state_restore_dict):
  state_dict = flax.serialization.to_state_dict(state)
  state_dict = copy_dict(state_dict, state_restore_dict)
  state = flax.serialization.from_state_dict(state, state_dict)

  return state


from pathlib import Path
from common.utils import load_json
from proj_configs import TRAIN_COLLECTION, VAL_COLLECTION, CHECKPOINTS_DIR_NAME
import imp
import ml_collections


def load_experiment(workdir, expm_cls=None, update_model_config={}, load_latest_ckpt=True, verbose=False):
  """

  :param workdir: e.g., 'train_xms/21965/mshyper-rd_lambda=0.08-latent_ch=320-base_ch=192'.
  :param expm_cls: if None, will use the 'main.py' saved in the workdir.
  :param update_model_config: if provided, will override the model_config saved in config.json.
  :return:
  """
  workdir = Path(workdir)

  if expm_cls is None:
    src_path = workdir / "main.py"
    expm_module = imp.load_source("main_exp", str(src_path))
    expm_cls = expm_module.Experiment

  cfg_path = workdir / "config.json"
  config = load_json(cfg_path)
  model_config = config["model_config"]
  model_config.update(update_model_config)

  if load_latest_ckpt:
    config['ckpt_restore_dir'] = str(workdir / TRAIN_COLLECTION / CHECKPOINTS_DIR_NAME)

  if verbose:
    logging.info("Will restore from %s", config['ckpt_restore_dir'])

  config = ml_collections.ConfigDict(config)
  expm = expm_cls(config)
  return expm


def load_ckpt_dict(workdir=None, ckpt_restore_dir=None):
  """
  Load ckpt dict saved by linen.
  :param workdir: e.g., 'train_xms/21965/mshyper-rd_lambda=0.08-latent_ch=320-base_ch=192'.
  :return:
  """
  if ckpt_restore_dir is None:
    assert workdir is not None
    workdir = Path(workdir)
    # Use latest.
    ckpt_restore_dir = str(workdir / TRAIN_COLLECTION / CHECKPOINTS_DIR_NAME)

  ckpt_restore = checkpoint.Checkpoint(ckpt_restore_dir)
  checkpoint_to_restore = ckpt_restore.get_latest_checkpoint_to_restore_from()
  assert checkpoint_to_restore
  state_restore_dict = ckpt_restore.restore_dict(checkpoint_to_restore)
  return state_restore_dict
