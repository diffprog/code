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


def gaussian_pdf(y, mu, sigma=1.0):
  t = y - mu
  return np.exp(-0.5 * (t/sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


def gaussian_cdf(y, mu, sigma=1.0):
  t = y - mu
  return 0.5 * (1 + erf(t / (np.sqrt(2) * sigma)))


def squared_loss(theta, y):
  return (y - theta) ** 2


ys = np.linspace(-5, 5, 100)
thetas = ys

plt.rcParams.update(
    {'lines.linewidth': 3,
    'legend.fontsize': 14,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 24,
    })

width = 20
height = 4
fig, axs = plt.subplots(1, 4, figsize=(4 * width, height))

axs[0].plot(ys, gaussian_pdf(mu=-2.0, y=ys), label=r"$\mu=-2$")
axs[0].plot(ys, gaussian_pdf(mu=0.0, y=ys), label=r"$\mu=0$")
axs[0].plot(ys, gaussian_pdf(mu=2.0, y=ys), label=r"$\mu=2$")

axs[0].set_xlabel(r"$y$")
axs[0].set_title(r"PDF")
axs[0].legend(loc="lower right")

axs[1].plot(ys, gaussian_cdf(mu=-2.0, y=ys), label=r"$\mu=-2$")
axs[1].plot(ys, gaussian_cdf(mu=0.0, y=ys), label=r"$\mu=0$")
axs[1].plot(ys, gaussian_cdf(mu=2.0, y=ys), label=r"$\mu=2$")
axs[1].set_xlabel(r"$y$")
axs[1].set_title(r"CDF $\mathbb{P}(Y \leq y)$")
axs[1].legend(loc="lower right")

axs[2].plot(thetas, thetas, c="k")
axs[2].set_xlabel(r"$\theta=\mu$")
axs[2].set_title(r"Mean function $A'(\theta)$")

axs[3].plot(thetas, squared_loss(theta=thetas, y=-2), label=r"$y=-2$")
axs[3].plot(thetas, squared_loss(theta=thetas, y=0), label=r"$y=0$")
axs[3].plot(thetas, squared_loss(theta=thetas, y=2), label=r"$y=2$")
axs[3].set_xlabel(r"$\theta$")
axs[3].set_title(r"Loss $L(\theta, y)$")
axs[3].legend(loc="upper right")

fig.set_tight_layout(True)

plt.show()
