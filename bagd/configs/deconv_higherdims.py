"""Deconvolution on a spherical source in various dimensions."""

import ml_collections


def d(**kwargs):
  """Helper of creating a config dict."""
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


seed = 0

data_dim = 2
def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.train_data_config = d(
    seed=seed,
    data_spec="sphere",
    data_dim=data_dim,
    batchsize=100000,
    gaussian_noise_var=0.1,
    fixed_batch=True  # Use one batch from the empirical distribution.
  )
  # config.eval_data_config = dict(reuse_train=True)
  config.eval_data_config = d(  # Use stochastic batches for eval.
    seed=seed,
    data_spec="sphere",
    data_dim=data_dim,
    batchsize=100000,
    gaussian_noise_var=0.1,
    fixed_batch=False,
  )
  config.train_eval_config = d(
    # epochs=100,
    # steps_per_epoch=10000,
    num_train_steps=10000,
    substeps=1,
    num_eval_steps=1,
    log_metrics_every_steps=10,
    log_imgs_every_steps=1000,
    checkpoint_every_steps=10000,
    eval_every_steps=500,
    seed=seed,
    # warm_start="",
    lr_decay_last_steps_ratio=1.0
  )

  config.model_config = d(
    # rd_lambda=2.0,
    rd_lambda=10.0,
    distort_type='half_sse',
    nu_support_size=1000,
  )
  config.optimizer_config = d(
    # name='adam',
    name='sgd',
    args=dict(),  # b1, b2, etc.
    learning_rate=1e-2,
    lr_decay=True,
    decay_type='inv_sqrt',
    decay_factor=0.1,
    # gradient_clip_norm=1.0,
  )
  config.ckpt_restore_dir = 'None'
  return config


def get_cfg_str(config):
  from collections import OrderedDict
  runname_dict = OrderedDict()
  # runname_dict['sig2'] = config.train_data_config.gaussian_noise_var
  runname_dict['d'] = config.train_data_config.data_dim
  runname_dict['n'] = config.model_config.nu_support_size
  runname_dict['rd_lambda'] = config.model_config.rd_lambda
  runname_dict['opt'] = config.optimizer_config.name
  runname_dict['tseed'] = config.train_eval_config.seed

  from common import utils
  return utils.config_dict_to_str(runname_dict, skip_falsy=False)


def get_hyper():
  """
  Produce a list of flattened dicts, each containing a hparam configuration overriding the one in
  get_config(), corresponding to one hparam trial/experiment/work unit.
  :return:
  """
  from common import hyper
  # gaussian_noise_vars = [0.1]
  # gaussian_noise_vars = hyper.sweep("train_data_config.gaussian_noise_var",
  #                                   gaussian_noise_vars)
  ds = [10, 5, 2]
  ds = hyper.izip(*[hyper.sweep(f"{split}_data_config.data_dim", ds) for split
          in ('train', 'eval')])
  # nu_support_sizes = [10, 100, 200, 1000, 2000]
  # nu_support_sizes = [10, 20, 50, 100, 300, 1000]
  nu_support_sizes = [1000, 500, 200, 100, 50, 20, 10]
  nu_support_sizes = hyper.sweep("model_config.nu_support_size",
                                 nu_support_sizes)
  # rd_lambdas = [0.01, 0.03, 0.1, 0.3, 1., 3., 10.]
  rd_lambdas = [10]
  rd_lambdas = hyper.sweep('model_config.rd_lambda', rd_lambdas)
  tseeds = hyper.sweep('train_eval_config.seed', list(range(5)))

  hparam_cfgs = hyper.product(ds, nu_support_sizes,
                              rd_lambdas, tseeds)
  return hparam_cfgs
