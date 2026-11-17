"""Exercise source loading, plotting, and the original training entry points."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from common.module_utils import load_module_from_path


ROOT = Path(__file__).resolve().parents[1]


def test_load_saved_module(tmp_path):
  source = tmp_path / 'saved.py'
  source.write_text('from __future__ import annotations\n'
                    'from dataclasses import dataclass\n'
                    '@dataclass\nclass Config:\n  value: int = 7\n')
  name = 'compatibility_saved_module'
  try:
    module = load_module_from_path(name, source)
    assert module.Config().value == 7
    source.write_text('raise RuntimeError("broken source")\n')
    with pytest.raises(RuntimeError, match='broken source'):
      load_module_from_path(name, source)
    assert sys.modules[name] is module
  finally:
    sys.modules.pop(name, None)


def test_plot_and_gif(tmp_path):
  from matplotlib.figure import Figure
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from PIL import Image
  from plot_utils import fig_to_np_arr, make_gif

  fig = Figure(figsize=(2, 1), dpi=50)
  FigureCanvasAgg(fig)
  fig.set_facecolor('red')
  pixels = fig_to_np_arr(fig, tight_layout=False)
  assert pixels.shape == (50, 100, 3)
  assert pixels.dtype == np.uint8
  np.testing.assert_array_equal(pixels[0, 0], [255, 0, 0])
  paths = [tmp_path / 'first.png', tmp_path / 'second.png']
  Image.fromarray(pixels).save(paths[0])
  Image.fromarray(255 - pixels).save(paths[1])
  output = tmp_path / 'animation.gif'
  make_gif(paths, output)
  with Image.open(output) as animation:
    assert animation.n_frames == 2
    for index in range(2):
      animation.seek(index)
      assert animation.info['duration'] == 500


@pytest.mark.parametrize('algorithm', ['ba', 'bagd', 'bahybrid'])
def test_training_entrypoint(algorithm, tmp_path):
  config = tmp_path / 'config.py'
  config.write_text('''from bagd.configs.deconv_higherdims import get_config as defaults, get_cfg_str

def get_config():
  config = defaults()
  config.train_data_config.batchsize = 32
  config.eval_data_config.batchsize = 32
  config.model_config.nu_support_size = 7
  config.train_eval_config.num_train_steps = 4
  config.train_eval_config.log_metrics_every_steps = 2
  config.train_eval_config.log_imgs_every_steps = 2
  config.train_eval_config.checkpoint_every_steps = 2
  config.train_eval_config.eval_every_steps = 2
  return config
''')
  env = dict(os.environ, MPLBACKEND='Agg', TF_NUM_INTRAOP_THREADS='1',
             TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1',
             XLA_PYTHON_CLIENT_PREALLOCATE='false')
  # Keep both invocations in the same experiment directory outside SLURM too.
  env.setdefault('SLURM_JOB_ID', 'training-test')
  command = [sys.executable, '-m', f'{algorithm}.train', f'--config={config}',
             f'--experiments_dir={tmp_path / "experiments"}', '--alsologtostderr']

  def run(args):
    result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, result.stdout
    return result.stdout

  run(command)
  workdirs = list((tmp_path / 'experiments').glob('*/*/config.json'))
  assert len(workdirs) == 1
  workdir = workdirs[0].parent
  for split in ['train', 'val']:
    records = [json.loads(line) for line in (workdir / split / 'record.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == [2, 4]
    assert all(np.isfinite(value) for record in records for value in record.values())
  assert len(list((workdir / 'train' / 'images').glob('*.png'))) == 2
  assert list((workdir / 'train' / 'images').glob('*.gif'))
  assert list((workdir / 'train').glob('events.out.tfevents.*'))

  run(command + ['--config.train_eval_config.num_train_steps=6'])
  records = [json.loads(line) for line in (workdir / 'val' / 'record.jsonl').read_text().splitlines()]
  assert [record['step'] for record in records] == [6]
  # Load the saved source and checkpoint, and evaluate the restored parameters.
  run([sys.executable, '-c', '''import sys
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from common.jax_experiment import load_experiment
experiment = load_experiment(sys.argv[1])
assert int(experiment.state.step) == 6
for value in experiment.state.params.values():
  assert np.isfinite(value).all()
np.testing.assert_allclose(np.asarray(experiment.state.params['nu_w']).sum(), 1., rtol=1e-5)
metrics = experiment.p_eval_step(experiment.state.params, jnp.asarray(next(experiment.eval_iter)))
assert all(np.isfinite(value).all() for value in metrics['scalars'].values())
''', str(workdir)])
