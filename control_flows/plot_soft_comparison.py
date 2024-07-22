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
from scipy.special import erf


def heaviside(x):
    return np.where(x > 0, 1, 0)


def logistic(x):
    return 1/(1 + np.exp(-x))


def gaussian_cdf(t, sigma=1.0):
  return 0.5 * (1 + erf(t / (np.sqrt(2) * sigma)))


def gaussian_kernel(t, sigma=1.0):
  return np.exp(-0.5 * (t/sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


def gaussian_eq(t):
  return gaussian_kernel(t) / gaussian_kernel(0)


def logistic_kernel(t):
  return np.exp(t) / ((1 + np.exp(t) ** 2))


def logistic_eq(t):
  return logistic_kernel(t) / logistic_kernel(0)


xs = np.linspace(-3, 3, 100)

plt.rcParams.update(
    {'lines.linewidth': 4,
    'legend.fontsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.titlesize': 24,
    })

fig, axs = plt.subplots(1, 2, figsize=(30, 4))

axs[0].plot([0, 0], [0, 1], c="C0", label="Hard")
axs[0].plot([-3, 3], [0, 0], c="C0")

for c, func, name, ls in zip(
  ["C1", "C2"],
  [logistic_eq, gaussian_eq],
  ['Logistic', 'Gaussian'],
  [None, "--"]):
    axs[0].plot(xs, func(xs), label=name, c=c, ls=ls)

axs[0].set_title('Soft equal zero')
axs[0].locator_params(axis='y', nbins=3)

for func, name, ls in zip([heaviside, logistic, gaussian_cdf],
                          ['Hard', 'Logistic', 'Gaussian'],
                          [None, None, "--"]):
    axs[1].plot(xs, func(xs), label=name, ls=ls)

axs[1].locator_params(axis='y', nbins=3)
axs[1].set_title('Soft greater than zero')
axs[1].legend(loc="lower right")

#plt.subplots_adjust()
plt.show()
