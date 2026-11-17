# Wasserstein GD on the entropic optimal transport (EOT) functional.
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
from functools import partial
from numbers import Integral
import common.jax_utils


sink_delta = 1e-6
max_sink_its = 1000


def _weights(w, n, column, dtype):
  """Accept vectors as well as the row/column convention used by bagd."""
  w = jnp.asarray(w)
  shape = (n, 1) if column else (1, n)
  if w.shape not in ((n,), shape):
    raise ValueError(f'Expected weights of shape {(n,)} or {shape}, got {w.shape}')
  if jnp.iscomplexobj(w):
    raise ValueError('Weights must be real')
  return w.astype(dtype).reshape(shape)


def _valid_weights(w):
  atol = max(1e-7, 8 * np.finfo(w.dtype).eps)
  return (jnp.all(jnp.isfinite(w)) & jnp.all(w > 0)
          & (jnp.abs(jnp.sum(w) - 1.) <= atol))


def _prepare_inputs(mu_x, mu_w, nu_x, nu_w, rd_lambda):
  mu_x, nu_x, rd_lambda = map(jnp.asarray, (mu_x, nu_x, rd_lambda))
  if (mu_x.ndim < 2 or mu_x.ndim != nu_x.ndim
      or mu_x.shape[1:] != nu_x.shape[1:]
      or 0 in mu_x.shape or 0 in nu_x.shape):
    raise ValueError('Points must have nonempty shapes (m, ...) and (n, ...) with matching features')
  if rd_lambda.ndim != 0:
    raise ValueError('rd_lambda must be a scalar')
  if any(jnp.iscomplexobj(v) for v in (mu_x, nu_x, rd_lambda)):
    raise ValueError('Points and rd_lambda must be real')
  dtype = jnp.result_type(mu_x, nu_x, jnp.asarray(mu_w), jnp.asarray(nu_w), rd_lambda, jnp.float32)
  mu_w = _weights(mu_w, mu_x.shape[0], True, dtype)
  nu_w = _weights(nu_w, nu_x.shape[0], False, dtype)
  return mu_x.astype(dtype), mu_w, nu_x.astype(dtype), nu_w, rd_lambda.astype(dtype)


def validate_inputs(mu_x, mu_w, nu_x, nu_w, rd_lambda):
  """Check finite inputs, positive normalized weights, and positive rd_lambda.

  Call outside jit to raise ValueError on invalid inputs. The compiled solver
  instead returns valid=False and a NaN objective.
  """
  mu_x, mu_w, nu_x, nu_w, rd_lambda = _prepare_inputs(mu_x, mu_w, nu_x, nu_w, rd_lambda)
  if not bool(_valid_weights(mu_w) & _valid_weights(nu_w)):
    raise ValueError('Weights must be finite, strictly positive, and normalized')
  if not bool(jnp.all(jnp.isfinite(mu_x)) & jnp.all(jnp.isfinite(nu_x))):
    raise ValueError('Points must be finite')
  if not bool(jnp.isfinite(rd_lambda) & (rd_lambda > 0)):
    raise ValueError('rd_lambda must be positive and finite')


@jax.jit
def sinkhorn_iterate(log_m1, log_m2, phi, psi, C):
  """One log-domain sweep; phi is (m, 1), psi is (1, n), C is (m, n)."""
  phi = -logsumexp(psi - C + log_m2, axis=1, keepdims=True)
  psi = -logsumexp(phi - C + log_m1, axis=0, keepdims=True)
  return phi, psi


@partial(jax.jit, static_argnames=['max_sink_its'])
def solve_sinkhorn(m1, m2, C, phi=None, psi=None, tol=sink_delta, max_sink_its=max_sink_its):
  """Solve EOT with unit entropy coefficient and an already scaled cost C.

  Returns (phi, psi, obj, scalar_metrics). Stopping uses the maximum of the two
  marginal L1 errors. On reaching max_sink_its, returns the last iterate with
  sinkhorn_converged=False. Invalid numerical inputs give valid=False,
  NaN obj, and zero iterations. max_sink_its must be a positive static integer.
  Optional warm-start potentials must be finite and shaped (m, 1)/(1, n).
  """
  C = jnp.asarray(C)
  if C.ndim != 2 or 0 in C.shape or jnp.iscomplexobj(C):
    raise ValueError('C must be a nonempty real matrix')
  if not isinstance(max_sink_its, Integral) or isinstance(max_sink_its, bool) or max_sink_its < 1:
    raise ValueError('max_sink_its must be a positive Python integer (static under jit)')
  dtype = jnp.result_type(C, jnp.asarray(m1), jnp.asarray(m2), jnp.float32)
  C = C.astype(dtype)
  m1, m2 = _weights(m1, C.shape[0], True, dtype), _weights(m2, C.shape[1], False, dtype)
  if any(p is not None and jnp.iscomplexobj(p) for p in (phi, psi)):
    raise ValueError('Potentials must be real')
  phi = jnp.zeros_like(m1) if phi is None else jnp.asarray(phi, dtype=dtype)
  psi = jnp.zeros_like(m2) if psi is None else jnp.asarray(psi, dtype=dtype)
  tol = jnp.asarray(tol, dtype=dtype)
  if phi.shape != m1.shape or psi.shape != m2.shape or tol.ndim != 0:
    raise ValueError('Potentials must have shapes (m, 1)/(1, n), and tol must be scalar')
  valid = (_valid_weights(m1) & _valid_weights(m2) & jnp.all(jnp.isfinite(C))
           & jnp.all(jnp.isfinite(phi)) & jnp.all(jnp.isfinite(psi))
           & jnp.isfinite(tol) & (tol > 0))
  log_m1, log_m2 = jnp.log(m1), jnp.log(m2)

  def cond_fun(carry):
    phi, psi, error, its = carry
    return (valid & (its < max_sink_its)
            & ((its == 0) | (jnp.isfinite(error) & (error > tol))))

  def body_fun(carry):
    phi, psi, _, its = carry
    phi, psi = sinkhorn_iterate(log_m1, log_m2, phi, psi, C)
    # Center the potentials without changing the coupling.
    phi_avg = jnp.mean(phi)
    phi, psi = phi - phi_avg, psi + phi_avg
    log_pi = log_m1 + log_m2 + phi + psi - C
    rows = jnp.exp(logsumexp(log_pi, axis=1, keepdims=True))
    cols = jnp.exp(logsumexp(log_pi, axis=0, keepdims=True))
    error = jnp.maximum(jnp.sum(jnp.abs(rows - m1)), jnp.sum(jnp.abs(cols - m2)))
    return phi, psi, error, its + 1

  init = phi, psi, jnp.asarray(jnp.inf, dtype=dtype), jnp.int32(0)
  phi, psi, error, its = jax.lax.while_loop(cond_fun, body_fun, init)
  log_pi = log_m1 + log_m2 + phi + psi - C
  # Full dual value. The correction vanishes when the coupling has mass one.
  obj = jnp.sum(phi * m1) + jnp.sum(psi * m2) - jnp.expm1(logsumexp(log_pi))
  finite = jnp.isfinite(obj) & jnp.isfinite(error)
  converged = valid & finite & (error <= tol)
  scalar_metrics = dict(sinkhorn_its=its, sinkhorn_error=jnp.where(valid, error, jnp.inf),
                        sinkhorn_converged=converged, valid=valid)
  return (jnp.where(valid, phi, jnp.nan), jnp.where(valid, psi, jnp.nan),
          jnp.where(valid, obj, jnp.nan), scalar_metrics)


@partial(jax.jit, static_argnames=['distort_type', 'max_sink_its'])
def wgrad_eot(mu_x, mu_w, nu_x, nu_w, distort_type: str, rd_lambda,
              tol=sink_delta, max_sink_its=max_sink_its):
  """Compute the Wasserstein gradient of the scaled EOT functional.

  loss = min_pi rd_lambda * <C, pi> + KL(pi || mu_w otimes nu_w), with both
  marginals fixed, i.e. rd_lambda * EOT with epsilon=1/rd_lambda. For unscaled
  EOT divide both loss and gradient by rd_lambda. This gradient has no nu_w factor;
  the ordinary coordinate derivative is nu_w[j] * gradient[j]. Check the
  sinkhorn_converged metric before using the gradient. Float32 is supported;
  enable JAX float64 for tight tolerances.
  """
  mu_x, mu_w, nu_x, nu_w, rd_lambda = _prepare_inputs(mu_x, mu_w, nu_x, nu_w, rd_lambda)
  pairwise_distort_fn = common.jax_utils.get_pairwise_distort_fn(distort_type)
  log_mu_w = jnp.log(mu_w)

  def compute_psi_sum(nu_x):
    C = pairwise_distort_fn(mu_x, nu_x)
    scaled_C = jnp.where(jnp.isfinite(rd_lambda) & (rd_lambda > 0), rd_lambda * C, jnp.nan)
    # Run Sinkhorn, without differentiating through the iterative solve.
    phi, psi, obj, metrics = solve_sinkhorn(mu_w, nu_w, jax.lax.stop_gradient(scaled_C),
                                          None, None, tol, max_sink_its)
    # Evaluate the second potential as a function of position, holding phi fixed.
    phi = jax.lax.stop_gradient(phi)
    psi = -logsumexp(phi - scaled_C + log_mu_w, axis=0, keepdims=True)
    psi_sum = jnp.sum(psi)
    return psi_sum, dict(loss=obj, **metrics)

  psi_prime, metrics = jax.grad(compute_psi_sum, has_aux=True)(nu_x)
  n = jnp.size(psi_prime, 0)
  psi_prime_norm = jnp.mean(jnp.linalg.norm(jnp.reshape(psi_prime, [n, -1]), axis=1))
  scalar_metrics = dict(**metrics, psi_prime_norm=psi_prime_norm)
  return psi_prime, scalar_metrics


@partial(jax.jit, static_argnames=['distort_type', 'max_sink_its'])
def estimate_eot(mu_x, mu_w, nu_x, nu_w, distort_type: str, rd_lambda,
                 tol=sink_delta, max_sink_its=max_sink_its):
  """Return (scalar_metrics, pi) for the fixed-marginal EOT problem.

  rate is KL(pi || mu otimes nu) in nats; distortion is the unscaled expected
  cost. loss is the scaled dual objective; eot_loss=loss/rd_lambda.
  Check sinkhorn_converged: primal_loss uses the returned coupling, which may
  not yet be feasible. For spatial derivatives, use wgrad_eot.
  """
  mu_x, mu_w, nu_x, nu_w, rd_lambda = _prepare_inputs(mu_x, mu_w, nu_x, nu_w, rd_lambda)
  pairwise_distort_fn = common.jax_utils.get_pairwise_distort_fn(distort_type)
  C = pairwise_distort_fn(mu_x, nu_x)
  scaled_C = jnp.where(jnp.isfinite(rd_lambda) & (rd_lambda > 0), rd_lambda * C, jnp.nan)
  phi, psi, obj, metrics = solve_sinkhorn(mu_w, nu_w, scaled_C, None, None, tol, max_sink_its)
  log_density = phi + psi - scaled_C
  pi = jnp.exp(jnp.log(mu_w) + jnp.log(nu_w) + log_density)
  distortion = jnp.sum(pi * C)
  rate = jnp.sum(pi * log_density)
  scalar_metrics = dict(loss=obj, eot_loss=obj / rd_lambda, rate=rate, distortion=distortion,
                        primal_loss=rd_lambda * distortion + rate, **metrics)
  return scalar_metrics, pi
