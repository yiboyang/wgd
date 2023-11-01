import numpy as np
import tensorflow as tf
from typing import NamedTuple, Mapping, Any, Sequence
from common import utils

class Metrics(NamedTuple):
  """
  A helper class to facilitate collecting/logging metrics. Adapted from my rdc/common/train_lib.py
  Also see https://github.com/google-research/google-research/blob/master/vct/src/metric_collection.py
  TODO: perhaps make it no longer depend on tf?
  """
  scalars: Mapping[str, Any]
  images: Mapping[str, tf.Tensor]

  # tensors: Mapping[str, tf.Tensor]
  @classmethod
  def make(cls):
    return Metrics(scalars={}, images={})

  def record_scalar(self, key, value):
    self.scalars[key] = value

  def record_scalars(self, scalars):
    # Inspired by writer_scalars https://github.com/google/CommonLoopUtils/blob/main/clu/metric_writers/summary_writer.py#L53
    for key, value in scalars.items():
      self.scalars[key] = value

  def record_image(self, key, value):
    self.images[key] = value

  # def record_tensor(self, key, val):
  #   self.tensors[key] = val

  @property
  def scalars_numpy(self):
    return {k: np.asarray(v).item() for (k, v) in self.scalars.items()}
    # return {k: v.numpy().item() for (k, v) in self.scalars.items()}

  @property
  def images_grid(self):
    return {k: utils.visualize_image_batch(v, crop_dim=256) for (k, v) in self.images.items()}

  @property
  def scalars_float(self):
    return {k: float(v) for (k, v) in self.scalars.items()}

  @classmethod
  def merge_metrics(cls, metrics_list):
    """
    :param metrics_list: a list of Metrics, where each metrics object is the output of the model on an input batch.
    :return:
    """
    # Reduce scalars by taking means.
    merged_scalars = {}
    scalars_keys = metrics_list[0].scalars.keys()
    for key in scalars_keys:
      merged_scalars[key] = tf.reduce_mean([m.scalars[key] for m in metrics_list])

    # Reduce images/tensors by concatenating across batches.
    merged_images = {}
    images_keys = metrics_list[0].images.keys()
    for key in images_keys:
      merged_images[key] = tf.concat([m.images[key] for m in metrics_list], axis=0)

    return Metrics(scalars=merged_scalars, images=merged_images)

