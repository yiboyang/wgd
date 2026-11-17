"""Load experiment and configuration modules from saved Python source."""

import importlib.util
import sys


def load_module_from_path(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  previous = sys.modules.get(name)
  sys.modules[name] = module
  try:
    spec.loader.exec_module(module)
  except Exception:
    if previous is None:
      del sys.modules[name]
    else:
      sys.modules[name] = previous
    raise
  return module
