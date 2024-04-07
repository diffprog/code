# Copyright 2023 Google LLC
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
from scipy.special import erf


def heaviside(t):
  return np.where(t >= 0, 1, 0)


def gaussian_cdf(t, sigma):
  return 0.5 * (1 + erf(t / (np.sqrt(2) * sigma)))


def relu(t):
  return np.maximum(t, 0)


def gaussian_kernel(t, sigma):
  return np.exp(-0.5 * (t/sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


def smoothed_relu(t, sigma):
  return (sigma ** 2 * gaussian_kernel(t, sigma) + t * gaussian_cdf(t, sigma))


t = np.linspace(-3, 3, 1000)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
ax1, ax2 = axes

ax1.plot(t, relu(t), lw=3)
ax1.plot(t, smoothed_relu(t, sigma=0.5), lw=3, label=r"$\sigma=0.5$")
ax1.plot(t, smoothed_relu(t, sigma=1.0), lw=3, label=r"$\sigma=1.0$")
ax1.plot(t, smoothed_relu(t, sigma=2.0), lw=3, label=r"$\sigma=2.0$")
ax1.legend(loc="best", fontsize=16)

ax2.plot(t, heaviside(t), lw=3)
ax2.plot(t, gaussian_cdf(t, sigma=0.5), lw=3)
ax2.plot(t, gaussian_cdf(t, sigma=1.0), lw=3)
ax2.plot(t, gaussian_cdf(t, sigma=2.0), lw=3)

plt.show()
