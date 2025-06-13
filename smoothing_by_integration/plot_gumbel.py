# Copyright 2025 Google LLC
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

import numpy as np
import matplotlib.pyplot as plt


gamma = np.euler_gamma


def sample(rng, size):
  """Sample from shifted standard Gumbel distribution."""
  u = rng.rand(size)
  return -np.log(-np.log(u)) - gamma


def pdf(z):
  """PDF of shifted standard Gumbel distribution."""
  nu = z + gamma + np.exp(-(z + gamma))
  return np.exp(-nu)


rng = np.random.RandomState(0)
z = sample(rng, 10000)
mean = np.mean(z)
mode = - gamma

z = np.linspace(-4, 4, 1000)

plt.figure()

plt.plot(z, pdf(z), lw=3, c="k", zorder=1)
plt.scatter([mean], [pdf(mean)], label="Mean", s=100, zorder=2)
plt.scatter([mode], [pdf(mode)], label="Mode", s=100, zorder=2)
plt.xlabel(r"$z$", size=16)
plt.ylabel(r"$p(z)$", size=16)
plt.title("Shifted standard Gumbel distribution", size=18)

plt.legend(loc="best", fontsize=16)

plt.tight_layout()

plt.show()
