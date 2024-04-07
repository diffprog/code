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
from matplotlib import pyplot as plt


def relu(x):
    return np.maximum(0, x)


def softplus(x):
    return np.log(1 + np.exp(x))


def sparseplus(x):
    return np.where(x <= -1, 0, np.where(x >= 1, x, (x + 1) ** 2 / 4))


def heaviside(x):
    return np.where(x > 0, 1, 0)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def sparsesigmoid(x):
    return np.where(x <= -1, 0, np.where(x >= 1, 1, (x + 1) / 2))


xs = np.linspace(-3, 3, 100)

plt.rcParams.update(
    {'lines.linewidth': 4,
    'legend.fontsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.titlesize': 24,
    })

fig, axs = plt.subplots(1, 2, figsize=(30, 4))

for func, name in zip([relu, softplus, sparseplus],
                      ['ReLU', 'Softplus', 'Sparseplus']):
  axs[0].plot(xs, func(xs), label=name)

axs[0].legend(loc='upper left')
axs[0].locator_params(axis='x', nbins=5)
axs[0].locator_params(axis='y', nbins=3)
axs[0].set_title('Activations')

for func, name in zip([heaviside, logistic, sparsesigmoid],
                      ['Heaviside', 'Logistic', 'Sparse\nsigmoid']):
  axs[1].plot(xs, func(xs), label=name)

axs[1].legend(loc='upper left')
axs[1].locator_params(axis='x', nbins=5)
axs[1].locator_params(axis='y', nbins=3)
axs[1].set_title('Sigmoids')

plt.show()
