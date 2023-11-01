import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
from functools import partial
import optax
from absl import logging


def mse(x, y):
  return jnp.mean((x - y) ** 2)


def sse(x, y):
  return jnp.sum((x - y) ** 2)


def half_sse(x, y):
  return 0.5 * jnp.sum((x - y) ** 2)


def get_pairwise_fun(fun):
  """
  Given a function, fun(x, y) -> scalar, return a new function that receives a batch of xs and a batch of ys and
  computes fun on all pairs of x in xs and y in ys.
  :param fun:
  :return: pairwise_fun: takes xs=[M, ...] and ys=[N, ...], returns an array of shape [M, N] whose [i,j]th entry
  corresponds to fun(xs[i], ys[j]).

  Inspired by https://stackoverflow.com/questions/69605121/multiple-vmap-in-jax
  Also see https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html
  """
  return jax.vmap(jax.vmap(fun, (None, 0)), (0, None))


# pairwise_mse_dy = get_pairwise_fun(jax.grad(mse, argnums=1))
def get_distort_fn(distort_type: str):
  if distort_type == 'mse':
    distort_fn = mse
  elif distort_type == 'sse':
    distort_fn = sse
  elif distort_type == 'half_sse':
    distort_fn = half_sse
  else:
    raise NotImplementedError
  return distort_fn


# def get_pairwise_distort_fn(distort_type: str):
#   distort_fn = get_distort_fn(distort_type)
#   return get_pairwise_fun(distort_fn)


# A more optimized version than the jax version above. Less elegant but should require much
# less memory and run faster too.
def get_pairwise_distort_fn(distort_type: str):
  def fun(xs, ys):
    return pairwise_distort_fn(xs, ys, distort_type)

  return fun


def pairwise_distort_fn(xs, ys, distort_type: str):
  # Convert to matrices.
  xs = jnp.reshape(xs, [jnp.size(xs, 0), -1])
  ys = jnp.reshape(ys, [jnp.size(ys, 0), -1])

  SE = pairwise_dist_squared(xs, ys)
  if distort_type in ('sse', 'se'):
    return SE
  elif distort_type == 'mse':
    data_dim = jnp.size(xs, 1)
    return SE / data_dim
  elif distort_type in ('hse', 'half_sse'):
    return SE * 0.5
  else:
    raise NotImplementedError


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
  na = jnp.sum(jnp.square(A), 1)
  nb = jnp.sum(jnp.square(B), 1)

  # na as a row and nb as a co"lumn vectors
  na = jnp.reshape(na, [-1, 1])
  nb = jnp.reshape(nb, [1, -1])

  # return pairwise squared euclidead difference matrix
  D = na - 2 * jnp.matmul(A, jnp.transpose(B)) + nb
  return D

