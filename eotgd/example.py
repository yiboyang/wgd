"""Minimal example of Wasserstein GD on the EOT functional.

Run from the project root: python -m eotgd.example --steps 100
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from eotgd.main import estimate_eot, validate_inputs, wgrad_eot


def run_demo(steps=100, seed=0, learning_rate=0.02, rd_lambda=10.,
             tol=1e-6, max_sink_its=1000, dtype=jnp.float32):
  """Noisy-circle example with fixed particle weights and Euler updates."""
  if steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
    raise ValueError('steps and learning_rate must be positive')
  rng = np.random.default_rng(seed)
  angles = rng.uniform(0, 2 * np.pi, size=256)
  mu_x = np.stack([np.cos(angles), np.sin(angles)], axis=1)
  mu_x += np.sqrt(0.1) * rng.normal(size=mu_x.shape)
  nu_x = mu_x[rng.choice(len(mu_x), 16, replace=False)].copy()
  mu_x, nu_x = jnp.asarray(mu_x, dtype), jnp.asarray(nu_x, dtype)
  mu_w = jnp.ones((len(mu_x), 1), dtype) / len(mu_x)
  nu_w = jnp.ones((1, len(nu_x)), dtype) / len(nu_x)
  validate_inputs(mu_x, mu_w, nu_x, nu_w, rd_lambda)

  def train_step(nu_x, unused):
    psi_prime, scalar_metrics = wgrad_eot(mu_x, mu_w, nu_x, nu_w, 'half_sse', rd_lambda,
                                         tol=tol, max_sink_its=max_sink_its)
    proposal = nu_x - learning_rate * psi_prime
    accepted = (scalar_metrics['sinkhorn_converged']
                & jnp.all(jnp.isfinite(proposal)) & jnp.all(jnp.isfinite(psi_prime)))
    new_nu_x = jnp.where(accepted, proposal, nu_x)
    scalar_metrics['step_accepted'] = accepted
    return new_nu_x, scalar_metrics

  train = jax.jit(lambda points: jax.lax.scan(train_step, points, None, length=steps))
  final_nu_x, history = train(nu_x)
  history = jax.device_get(history)
  if not np.all(history['step_accepted']):
    failed = int(np.flatnonzero(~history['step_accepted'])[0])
    raise RuntimeError(f'Step {failed} rejected: Sinkhorn error={history["sinkhorn_error"][failed]:.3g}, '
                       f'iterations={history["sinkhorn_its"][failed]}. '
                       'Increase max_sink_its, relax tol, or use --x64 for tight tolerances.')
  final_metrics, _ = estimate_eot(mu_x, mu_w, final_nu_x, nu_w, 'half_sse', rd_lambda,
                                   tol=tol, max_sink_its=max_sink_its)
  final_metrics = jax.device_get(final_metrics)
  if not final_metrics['sinkhorn_converged']:
    raise RuntimeError('Final evaluation did not converge; adjust Sinkhorn settings.')
  summary = dict(initial_loss=float(history['loss'][0]), final_loss=float(final_metrics['loss']),
                 final_eot_loss=float(final_metrics['eot_loss']),
                 max_marginal_error=float(max(history['sinkhorn_error'].max(), final_metrics['sinkhorn_error'])),
                 max_sinkhorn_its=int(max(history['sinkhorn_its'].max(), final_metrics['sinkhorn_its'])),
                 steps=steps, rd_lambda=rd_lambda, learning_rate=learning_rate,
                 tol=tol, dtype=str(mu_x.dtype))
  arrays = dict(mu_x=np.asarray(mu_x), mu_w=np.asarray(mu_w), nu_w=np.asarray(nu_w),
                initial_nu_x=np.asarray(nu_x), final_nu_x=np.asarray(final_nu_x),
                loss=np.append(history['loss'], final_metrics['loss']),
                sinkhorn_error=history['sinkhorn_error'], sinkhorn_its=history['sinkhorn_its'])
  return summary, arrays


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--steps', type=int, default=100)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--learning-rate', type=float, default=0.02)
  parser.add_argument('--rd-lambda', type=float, default=10.)
  parser.add_argument('--tol', type=float, default=1e-6)
  parser.add_argument('--max-sink-its', type=int, default=1000)
  parser.add_argument('--x64', action='store_true')
  parser.add_argument('--output', type=Path, help='Optional .npz output for particles and loss history')
  args = parser.parse_args()
  if args.x64:
    jax.config.update('jax_enable_x64', True)
  summary, arrays = run_demo(args.steps, args.seed, args.learning_rate, args.rd_lambda,
                             args.tol, args.max_sink_its, jnp.float64 if args.x64 else jnp.float32)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
