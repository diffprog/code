# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gradient estimation of perturbed blackbox functions.
"""

import numpy as np
import matplotlib.pyplot as plt


def sfe1(f, mu, sigma, n_samples, rng):
  """Score function estimator (SFE)."""
  # Equivalent to:
  # samples = rng.normal(loc=mu, scale=sigma, size=n_samples)
  # values = [f(u) * (u - mu) / (sigma ** 2) for u in samples]
  samples = rng.normal(size=n_samples)
  values = [f(mu + sigma * z) * z / sigma for z in samples]
  return np.mean(values)


def sfe2(f, mu, sigma, n_samples, rng):
  """SFE with variance reduction, aka evolution strategies."""
  samples = rng.normal(size=n_samples)
  values = [(f(mu + sigma * z) - f(mu)) * z / sigma for z in samples]
  return np.mean(values)


def sfe3(f, mu, sigma, n_samples, rng):
  """SFE with central difference."""
  samples = rng.normal(size=n_samples)
  values = [(f(mu + sigma * z) - f(mu - sigma * z)) * z / (2*sigma) for z in samples]
  return np.mean(values)


def error(sfe, f, fp, mu, sigma, n_samples, rng, n_runs):
  errors = np.zeros(n_runs)
  for i in range(n_runs):
    errors[i] = np.abs(sfe(f, mu, sigma, n_samples=n_samples, rng=rng) - fp)
  return np.mean(errors)


if __name__ == '__main__':
  def f(u):
    return u ** 3

  def f_prime(u):
    return 3 * u ** 2

  mu = 3.5
  sigma = 0.1
  fp = f_prime(mu)

  sample_range = np.logspace(0, 7, 20)
  errors1 = np.zeros(len(sample_range))
  errors2 = np.zeros(len(sample_range))
  errors3 = np.zeros(len(sample_range))
  n_runs = 5
  rng = np.random.RandomState(0)

  for i, n_samples in enumerate(sample_range):
    n_samples = int(n_samples)
    errors1[i] = error(sfe1, f, fp, mu, sigma, n_samples, rng, n_runs)
    errors2[i] = error(sfe2, f, fp, mu, sigma, n_samples, rng, n_runs)
    errors3[i] = error(sfe3, f, fp, mu, sigma, n_samples, rng, n_runs)

  plt.figure()
  plt.plot(sample_range, errors1, label="SFE", lw=3)
  plt.plot(sample_range, errors2, label="SFE with forward difference", lw=3)
  plt.plot(sample_range, errors3, label="SFE with central difference", lw=3)
  plt.legend(loc="best", fontsize=16)
  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel("Number of samples", fontsize=16)
  plt.ylabel("Gradient error", fontsize=16)
  plt.show()
