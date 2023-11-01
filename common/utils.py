# Util methods for running experiments. TODO: merge with common_utils.py
import tensorflow as tf
import os
import sys
import json


def pairwise_dist_squared(A, B):
  """
  Computes pairwise distances between each elements of A and each elements of B.
  Optimized version, from https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
  Also see https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
  Args:
    A,    [m,d] matrix
    B,    [n,d] matrix
  Returns:
    D,    [m,n] matrix of pairwise distances
  """
  # squared norms of each row in A and B
  na = tf.reduce_sum(tf.square(A), 1)
  nb = tf.reduce_sum(tf.square(B), 1)

  # na as a row and nb as a co"lumn vectors
  na = tf.reshape(na, [-1, 1])
  nb = tf.reshape(nb, [1, -1])

  # return pairwise squared euclidead difference matrix
  # D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
  D = na - 2 * tf.matmul(A, B, False, True) + nb
  return D


def get_keras_optimizer(name):
  name_to_optimizer = {
    'sgd': tf.keras.optimizers.SGD,
    'adadelta': tf.keras.optimizers.Adadelta,
    'rmsprop': tf.keras.optimizers.RMSprop,
    'adam': tf.keras.optimizers.Adam
  }
  return name_to_optimizer[name]


def load_json(path):
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)


def dump_json(obj, path):
  with tf.io.gfile.GFile(path, 'w') as f:
    return json.dump(obj, f, indent=2)


class ClassBuilder(dict):
  """
  Example:
    class A:
      def __init__(self, arg1, arg2):
        ...

  class_builder = ClassBuilder(ClassA=A)
  ClassBuilder('ClassA', arg1='x', arg2='y') -> A('x', 'y')
  """

  def build(self, class_name, **kwargs):
    cls = self[class_name]
    return cls(**kwargs)


# Below methods help manage experiments.

try:
  from proj_configs import args_abbr
except:
  args_abbr = {}


def config_dict_to_str(cfg, record_keys=None, skip_falsy=True, prefix=None, args_abbr=args_abbr,
                       primary_delimiter='-', secondary_delimiter='_'):
  """
  Given a dictionary of cmdline arguments, return a string that identifies the training run.
  This is really pretty much just a copy of the config_dict_to_str from common_utils.py.
  :param cfg:
  :param record_keys: an iterable of strings corresponding to the keys to record. Default (None) is
  to record every (k,v) pair in the given dict.
  :param skip_falsy: whether to skip keys whose values evaluate to falsy (0, None, False, etc.)
  :param use_abbr: whether to use abbreviations for long key name
  :param primary_delimiter: the char to delimit different key-value paris
  :param secondary_delimiter: the delimiter within each key or value string (e.g., when the value is a list of numbers)
  :return:
  """
  kv_strs = []  # ['key1=val1', 'key2=val2', ...]
  if record_keys is None:  # Use all keys.
    record_keys = iter(cfg)
  for key in record_keys:
    val = cfg[key]
    if skip_falsy and not val:
      continue

    if isinstance(val, (list, tuple)):  # e.g., 'num_layers: [10, 8, 10] -> 'num_layers=10_8_10'
      val_str = secondary_delimiter.join(map(str, val))
    else:
      val_str = str(val)

    if args_abbr:
      key = args_abbr.get(key, key)

    kv_strs.append('%s=%s' % (key, val_str))

  if prefix:
    substrs = [prefix] + kv_strs
  else:
    substrs = kv_strs
  return primary_delimiter.join(substrs)


def get_xid():
  # See https://slurm.schedmd.com/job_array.html#env_vars
  xid = os.environ.get("SLURM_ARRAY_JOB_ID", None)
  if xid:
    return xid
  xid = os.environ.get("SLURM_JOB_ID", None)
  if xid:
    return xid
  return get_time_str()


def get_wid():
  return os.environ.get("SLURM_ARRAY_TASK_ID", None)


def get_run_info():
  run_info = {}
  run_info['cmdline'] = " ".join(
    sys.argv)  # attempt to reconstruct the original cmdline; not reliable (e.g., loses quotes)
  run_info['most_recent_version'] = get_git_revision_short_hash()

  for env_var in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID"):  # (xid, wid)
    if env_var in os.environ:
      run_info[env_var] = os.environ[env_var]

  import socket
  run_info['host_name'] = socket.gethostname()
  return run_info


def log_run_info(workdir):
  run_info = get_run_info()
  with open(os.path.join(workdir, f"run_info.json"), "w") as f:
    json.dump(run_info, f, indent=2)



from common.common_utils import preprocess_float_dict

import numpy as np
import tensorflow as tf


def get_time_str(strftime_format="%Y,%m,%d,%H%M%S"):
  import datetime
  if not strftime_format:
    from proj_configs import strftime_format

  time_str = datetime.datetime.now().strftime(strftime_format)
  return time_str


def psnr_to_float_mse(psnr):
  return 10 ** (-psnr / 10)


def float_mse_to_psnr(float_mse):
  return -10 * np.log10(float_mse)


# My custom logging code for logging in JSON lines ("jsonl") format
import json


class MyJSONEncoder(json.JSONEncoder):
  # https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return super(MyJSONEncoder, self).default(obj)


def get_json_logging_callback(log_file_path, buffering=1, **preprocess_float_kwargs):
  # Modified JSON logger example from https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback
  # Default is minimal buffering (=1)
  log_file = open(log_file_path, mode='wt', buffering=buffering)
  json_logging_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs_dict: log_file.write(
      json.dumps({'epoch': epoch, **preprocess_float_dict(logs_dict, **preprocess_float_kwargs)},
                 cls=MyJSONEncoder) + '\n'),
    on_train_end=lambda logs: log_file.close()
  )
  return json_logging_callback


# https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
import subprocess


def get_git_revision_hash() -> str:
  return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_revision_short_hash() -> str:
  return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


""" Run with temporary verbosity """


# from VDM code base

def with_verbosity(temporary_verbosity_level, fn):
  from absl import logging
  old_verbosity_level = logging.get_verbosity()
  logging.set_verbosity(temporary_verbosity_level)
  result = fn()
  logging.set_verbosity(old_verbosity_level)
  return result
