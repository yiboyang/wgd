import tensorflow as tf
import numpy as np
from typing import Optional, Callable


def read_png(filename, channels=3):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=channels)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)


def quantize_image(image):
  return tf.saturate_cast(tf.round(image), tf.uint8)


def center_crop_image(image, target_height, target_width):
  # Based on https://github.com/keras-team/keras/blob/v2.10.0/keras/layers/preprocessing/image_preprocessing.py#L202
  input_shape = tf.shape(image)
  H_AXIS = -3
  W_AXIS = -2
  h_diff = input_shape[H_AXIS] - target_height
  w_diff = input_shape[W_AXIS] - target_width

  tf.debugging.assert_greater_equal(h_diff, 0)
  tf.debugging.assert_greater_equal(w_diff, 0)

  h_start = tf.cast(h_diff / 2, tf.int32)
  w_start = tf.cast(w_diff / 2, tf.int32)
  return tf.image.crop_to_bounding_box(image, h_start, w_start, target_height, target_width)


def mse_psnr(x, y, max_val=255.):
  """Compute MSE and PSNR b/w two image tensors."""
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.float32)

  squared_diff = tf.math.squared_difference(x, y)
  axes_except_batch = list(range(1, len(squared_diff.shape)))

  # Results have shape [batch_size]
  mses = tf.reduce_mean(tf.math.squared_difference(x, y), axis=axes_except_batch)  # per img
  # psnrs = -10 * (np.log10(mses) - 2 * np.log10(255.))
  psnrs = -10 * (tf.math.log(mses) - 2 * tf.math.log(max_val)) / tf.math.log(10.)
  return mses, psnrs


