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

import numpy as np
import matplotlib.pyplot as plt


def moreau_env(mu, f, u):
  """
  Compute the Moreau envelope by grid search.

  mu: input values
  f: array containing the values of f.
  u: grid on which f has been evaluated.

  One should ensure that u is a superset of mu.

  Returns: an array of the same size has mu.
  """
  D = 0.5 * np.subtract.outer(mu, u) ** 2
  return np.min(f + D, axis=1)


def relu(u):
  return np.maximum(u, 0)


def ramp(u):
  return np.where(u >= 1, 1, relu(u))


def step(u):
  return np.where(u >= 0, 1, 0)


mu = np.linspace(-3, 3, 1000)
u = np.linspace(-5, 5, 10000)

plt.rcParams.update(
    {'lines.linewidth': 4,
    'legend.fontsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.titlesize': 24,
    })

fig, axs = plt.subplots(1, 3, figsize=(45, 4))

axs[0].plot(mu, relu(mu), label='Original', color='C2')
axs[0].plot(mu, moreau_env(mu, relu(u), u), label='Moreau env', ls='--',
            color='C4')
axs[0].set_title('ReLU')
axs[0].legend(loc='upper left')

axs[1].plot(mu, ramp(mu), color='C2')
axs[1].plot(mu, moreau_env(mu, ramp(u), u), ls='--', color='C4')
axs[1].set_title('Ramp')

axs[2].plot(mu, step(mu), color='C2')
axs[2].plot(mu, moreau_env(mu, step(u), u), ls='--', color='C4')
axs[2].set_title('Step')

fig.set_tight_layout(True)

plt.show()
