import tensorflow as tf
import numpy as np
import proj_configs
from common import image_utils
from common.image_utils import read_png, quantize_image
import glob
import math




def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def normalize_image(image):
  return image / 255. - 0.5


def unnormalize_image(float_tensor):
  return (float_tensor + 0.5) * 255.


def process_image(image, crop=None, patchsize=None, normalize=True, image_channels=3, augment=False):
  """
  Convert uint8 image into floating type appropriate for training, with optional cropping/augmentation.
  :param image:
  :param crop:
  :param patchsize:
  :param normalize:
  :param image_channels:
  :param augment:
  :return:
  """
  if crop is not None:
    assert patchsize > 0
    if crop == "random":
      image = tf.image.random_crop(image, (patchsize, patchsize, image_channels))
    elif crop == "center":
      image = image_utils.center_crop_image(image, patchsize, patchsize)
    else:
      raise NotImplementedError(crop)

  image = tf.cast(image, tf.float32)
  if normalize:
    image = normalize_image(image)
  if augment:
    image = augment_image(image)
  return image


def floats_to_pixels(x, training):
  x = unnormalize_image(x)
  if not training:
    x = quantize_image(x)
  return x


def augment_image(image):
  # also maybe apply random rotation
  return tf.image.random_flip_left_right(image, seed=None)


def get_image_dataset_from_tfds(name, split, shuffle, repeat, drop_remainder, batchsize, crop=None, patchsize=None,
                                normalize=True,
                                augment_imgs=False):
  """Creates input data pipeline from a TF Datasets dataset.
  :param repeat:
  """
  import tensorflow_datasets as tfds
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=shuffle)
    if split == 'test' and shuffle:
      print('Loaded test split with shuffle=True; you may want to use False instead for running evaluation.')
    # if split == "train":
    if repeat:
      dataset = dataset.repeat()
    img_channels = 3
    if patchsize is not None:  # filter out imgs smaller than patchsize (if not using full-sized images)
      if 'cifar' in name:
        assert patchsize <= 32
      elif 'mnist' in name:  # FYI tfds MNIST dataset has image shape [28, 28, 1]
        assert patchsize <= 28
        img_channels = 1
      else:
        dataset = dataset.filter(
          lambda x: check_image_size(x["image"], patchsize))
    dataset = dataset.map(
      lambda x: process_image(x["image"], crop=crop, patchsize=patchsize, normalize=normalize,
                              image_channels=img_channels, augment=augment_imgs))
    dataset = dataset.batch(batchsize, drop_remainder=drop_remainder)
  return dataset


# for reading images in .npy format
def read_npy_file_helper(file_name_in_bytes):
  # data = np.load(file_name_in_bytes.decode('utf-8'))
  data = np.load(file_name_in_bytes)  # turns out this works too without decoding to str first
  # assert data.dtype is np.float32   # needs to match the type argument in the caller tf.data.Dataset.map
  return data


def get_image_dataset_from_glob(file_glob, shuffle, repeat, drop_remainder, batchsize, crop=None, patchsize=None,
                                normalize=True, preprocess_threads=16):
  """Creates input data pipeline from custom PNG images. Formerly named 'get_custom_dataset'.
  :param file_glob:
  :param args:
  """
  import glob
  with tf.device("/cpu:0"):
    files = sorted(glob.glob(file_glob))
    if not files:
      raise RuntimeError(f"No images found with glob '{file_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
      dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if repeat:
      dataset = dataset.repeat()

    # if '.npy' in args.train_glob:  # reading numpy arrays directly instead of from images
    #    dataset = dataset.map(  # https://stackoverflow.com/a/49459838
    #        lambda item: tuple(tf.numpy_function(read_npy_file_helper, [item], [tf.float32, ])),
    #        num_parallel_calls=args.preprocess_threads)
    # else:
    #    dataset = dataset.map(
    #        read_png, num_parallel_calls=args.preprocess_threads)
    # dataset = dataset.map(lambda x: crop_image(x, args.patchsize))
    if '.npy' in file_glob:  # reading numpy arrays directly instead of from images
      dataset = dataset.map(  # https://stackoverflow.com/a/49459838
        lambda file_name: tuple(tf.numpy_function(read_npy_file_helper, [file_name], [tf.float32, ])),
        num_parallel_calls=preprocess_threads)
      dataset = dataset.map(lambda x: process_image(x, crop=crop, patchsize=patchsize, normalize=normalize),
                            num_parallel_calls=preprocess_threads)
    else:
      dataset = dataset.map(
        lambda x: process_image(read_png(x), crop=crop, patchsize=patchsize, normalize=normalize),
        num_parallel_calls=preprocess_threads)

    dataset = dataset.batch(batchsize, drop_remainder=drop_remainder)
  return dataset




TOY_SOURCE_NAMES = ['gaussian', 'banana', 'sphere'] + list(proj_configs.biggan_class_names_to_ids.keys())
IMG_SOURCE_NAMES = ['mnist', 'cifar', 'cifar10'] + list(proj_configs.biggan_class_names_to_ids.keys())


def get_toy_dataset(name, data_dim: int, batchsize: int, dtype='float32', **kwargs):
  # Create dataset pipeline in tensorflow (graph mode).
  assert name in TOY_SOURCE_NAMES
  if kwargs.get('seed') is not None:
    tf.random.set_seed(kwargs['seed'])

  if name == 'gaussian':
    if kwargs.get('gparams_path'):
      gparams = np.load(kwargs['gparams_path'])
      loc = gparams['loc'].astype(dtype)
      scale = gparams['scale'].astype(dtype)
      assert len(loc) == data_dim, 'Mismatch between gparams and requested data_dim'
    else:
      loc = np.zeros(data_dim, dtype=dtype)
      scale = np.ones(data_dim, dtype=dtype)
    # import tensorflow_probability
    # source = tensorflow_probability.distributions.Normal(loc=loc, scale=scale)
    # map_sample_fun = lambda _: source.sample(batchsize)
    map_sample_fun = lambda _: tf.random.normal([batchsize, data_dim], mean=loc, stddev=scale)

  elif name == 'sphere':  # Samples from the unit sphere.
    def sample_sphere(batchsize, ndim):
      V = tf.random.normal([ndim, batchsize], dtype=dtype)
      V /= tf.linalg.norm(V, axis=0)
      return tf.transpose(V)

    map_sample_fun = lambda _: sample_sphere(batchsize, data_dim)

  elif name == 'banana':
    from common import ntc_sources
    source = ntc_sources.get_banana()  # a tfp.distributions.TransformedDistribution object
    if data_dim == 2:
      map_sample_fun = lambda _: source.sample(batchsize)
    else:
      from common.ntc_sources import get_nd_banana
      map_sample_fun, _ = get_nd_banana(data_dim, kwargs['embedder_depth'], batchsize, kwargs.get('seed', 0))

  elif name in proj_configs.biggan_class_names_to_ids:
    # with tf.device("/gpu:0"):
    from common import biggan  # takes a while to import/setup; that's why I'm doing lazy import
    # flatten = tf.keras.layers.Flatten()
    # post_process_fun = lambda x: flatten(
    #   0.5 * x)  # map from [-1, 1] to [-0.5, 0.5], then flatten to [batch, npixels].
    sampler = biggan.get_sampler(name, data_dim)

    def map_sample_fun(_):
      # Sample each batch in small chunks then combine (otherwise can get OOM on GPU).
      # We rely on tf autograph to convert this method into graph mode.
      biggan_chunksize = min(batchsize, 32)  # this seems to work well on our Titan RTX GPU
      chunks = []
      for i in range(batchsize // biggan_chunksize + 1):
        chunk = sampler(biggan_chunksize)
        chunks.append(chunk)
      x = tf.concat(chunks, axis=0)
      x = x[:batchsize]
      x *= 0.5  # Convert from from [-1, 1] to [-0.5, 0.5]
      return x

  else:
    raise NotImplementedError

  dataset = tf.data.Dataset.from_tensors(
    [])  # got this trick from https://github.com/tensorflow/compression/blob/66228f0faf9f500ffba9a99d5f3ad97689595ef8/models/toy_sources/compression_model.py#L121
  dataset = dataset.repeat()
  dataset = dataset.map(map_sample_fun, num_parallel_calls=kwargs.get('preprocess_threads', 8))

  return dataset


def get_dataset_from_tensor_slices(tensor_slices, shuffle=False, first_N=None, mode='repeat', **kwargs):
  """
  Given some abstract iterable of tensor slices, return a custom dataset made of them.
  :param tensor_slices:
  :param shuffle:
  :param first_N:
  :param mode:
  :param kwargs: If kwargs['batchsize'] is unspecified (or None), the resulting dataset is unbatched, i.e., iterating
  over it returns individual tensors. If provided an int, the usual semantics apply; if provided the special string
  "all", the batchsize will be set equal to len(tensor_slices) (fortunately, unaffected by first_N).
  :return:
  """
  assert mode in ('swrep', 'repeat', 'once')
  if (seed := kwargs.get('seed')) is not None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

  if shuffle:
    np.random.shuffle(tensor_slices)

  if first_N is not None and first_N > 0:  # Only use the first this many samples from X.
    tensor_slices = tensor_slices[:first_N]

  if mode == 'swrep':
    batchsize = kwargs.get('batchsize')
    assert batchsize, "Must specify batchsize when sampling with replacement."
    dataset = tf.data.Dataset.from_tensors([])
    dataset = dataset.repeat()

    def map_sample_fun(_):
      # Take random elements from X with replacement.
      rand_idx = tf.random.uniform(shape=[batchsize], minval=0, maxval=len(tensor_slices), dtype=tf.int32)
      batch = tf.gather(tensor_slices, rand_idx)
      return batch

    dataset = dataset.map(map_sample_fun)
  else:
    dataset = tf.data.Dataset.from_tensor_slices(tensor_slices)
    if mode == 'repeat':
      dataset = dataset.repeat()
    # else, it's a one-off finite dataset

    if (batchsize := kwargs.get('batchsize')) is not None:
      if batchsize == 'all':
        batchsize = len(tensor_slices)

      dataset = dataset.batch(batchsize)

  return dataset


def get_np_dataset(np_path, shuffle=False, first_N=None, mode='repeat', dtype='float32', **kwargs):
  assert np_path.endswith('.npy') or np_path.endswith('.npz')
  X = np.load(np_path)
  X = X.astype(dtype)
  if kwargs.get('append_channel_dim'):  # convolutional models often require data to have a channel dim
    X = X[..., np.newaxis]
  return get_dataset_from_tensor_slices(X, shuffle=shuffle, first_N=first_N, mode=mode, **kwargs)


# Unified data loader.
def get_dataset(data_spec: str, gaussian_noise_var=None, **kwargs):
  if data_spec in TOY_SOURCE_NAMES:
    dataset = get_toy_dataset(data_spec, **kwargs)
  elif data_spec.endswith('.npy') or data_spec.endswith('.npz'):
    dataset = get_np_dataset(data_spec, **kwargs)
    if 'mnist' in data_spec:  # Do extra processing for imgs
      dataset = dataset.map(
        lambda x: process_image(x, crop=kwargs.get('crop'), patchsize=kwargs.get('patchsize'),
                                normalize=True, image_channels=1, augment=kwargs.get('augment')),
        num_parallel_calls=kwargs.get('preprocess_threads'))
  else:  # Custom images.
    if data_spec in proj_configs.dataset_to_globs.keys():
      file_glob = proj_configs.dataset_to_globs[data_spec]
    else:
      file_glob = data_spec

    with tf.device("/cpu:0"):
      paths = sorted(glob.glob(file_glob))
      if not paths:
        raise RuntimeError(f"No images found with glob '{file_glob}'.")
      paths_ds = get_dataset_from_tensor_slices(paths, shuffle=True, **kwargs)
      dataset = paths_ds.map(
        lambda x: process_image(read_png(x), crop=kwargs.get('crop'), patchsize=kwargs.get('patchsize'),
                                normalize=True, augment=kwargs.get('augment')),
        num_parallel_calls=kwargs.get('preprocess_threads'))

  if gaussian_noise_var:
    dataset = dataset.map(
      lambda x: x + tf.random.normal(shape=tf.shape(x), stddev=math.sqrt(gaussian_noise_var), dtype=x.dtype))

  return dataset
