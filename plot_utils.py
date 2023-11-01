# Quick tools for plotting.

import os
import pandas as pd
import numpy as np


def load_jsonl(jsonl, to_df=True):
  import json_lines
  with open(jsonl, 'r') as f:
    records = list(json_lines.reader(f))
  if to_df:
    return pd.DataFrame(records)
  return records


def plot_jsonl(jsonl, ax=None, x_key='step', y_key='loss', itv=1, dropna=True, **kwargs):
  import matplotlib.pyplot as plt
  df = load_jsonl(jsonl, to_df=True)
  df = df.set_index(x_key)

  if ax is None:
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', None))
  label = kwargs.pop('label', y_key)
  title = kwargs.pop('title', None)
  y_series = df[y_key]
  if dropna:
    y_series = y_series.dropna()
  ax.plot(y_series[::itv], label=label, **kwargs)
  # [ax.set_xlim(xlim) for ax in axs]
  # [ax.set_ylim(ylim) for ax in axs]
  plt.suptitle(title)
  return ax


def fig_to_np_arr(fig, tight_layout=True):
  """
  Convert a matplotlib figure into an RGB np array. https://stackoverflow.com/a/7821917
  :param fig:
  :return:
  """
  import numpy as np
  if tight_layout:
    fig.tight_layout(pad=0)
  fig.canvas.draw()

  # Now we can save it to a numpy array.
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data


# From https://github.com/yiboyang/bounding-rd/blob/5661750a979e3b4674c620ee49b58db847b19262/plot_utils.py
def natural_sort(l):
  # https://stackoverflow.com/a/4836734
  import re
  convert = lambda text: int(text) if text.isdigit() else text.lower()
  alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
  return sorted(l, key=alphanum_key)


def make_gif(imgs_paths, output_gif_path, sort_paths=False, duration_per_img=0.5, **kwargs):
  import imageio
  imgs = []
  if sort_paths:
    imgs_paths = natural_sort(imgs_paths)
  for file_path in imgs_paths:
    imgs.append(imageio.imread(file_path))
  return imageio.mimsave(output_gif_path, imgs, duration=duration_per_img, **kwargs)

