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
from scipy.special import factorial


def poisson_pdf(lam, y):
  return lam ** y * np.exp(-lam) / factorial(y)


def poisson_cdf(lam, y):
  return np.cumsum(poisson_pdf(lam, y))


def poisson_mean(theta):
  return np.exp(theta)


def poisson_loss(theta, y):
  return -y * theta + np.exp(theta) + np.log(factorial(y))


ys = np.arange(21)
thetas = np.linspace(-3, 3, 100)

plt.rcParams.update(
    {'lines.linewidth': 3,
    'legend.fontsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 24,
    })

width = 20
height = 4
fig, axs = plt.subplots(1, 4, figsize=(4 * width, height))

axs[0].plot(ys, poisson_pdf(lam=1.0, y=ys), lw=1, marker="s",
            label=r"$\lambda=1$")
axs[0].plot(ys, poisson_pdf(lam=4.0, y=ys), lw=1, marker="s",
            label=r"$\lambda=4$")
axs[0].plot(ys, poisson_pdf(lam=10.0, y=ys), lw=1, marker="s",
            label=r"$\lambda=10$")
axs[0].set_xlabel(r"$y$")
axs[0].set_title(r"PMF $\mathbb{P}(Y = y)$")
axs[0].legend(loc="upper right")

axs[1].plot(ys, poisson_cdf(lam=1.0, y=ys), lw=1, marker="s",
            label=r"$\lambda=1$")
axs[1].plot(ys, poisson_cdf(lam=4.0, y=ys), lw=1, marker="s",
            label=r"$\lambda=4$")
axs[1].plot(ys, poisson_cdf(lam=10.0, y=ys), lw=1, marker="s",
            label=r"$\lambda=10$")
axs[1].set_xlabel(r"$y$")
axs[1].set_title(r"CDF $\mathbb{P}(Y \leq y)$")
axs[1].legend(loc="lower right")

axs[2].plot(thetas, poisson_mean(thetas), c="k")
axs[2].set_xlabel(r"$\theta$")
axs[2].set_title(r"Mean function $A'(\theta)$")

axs[3].plot(thetas, poisson_loss(theta=thetas, y=1.0), label=r"$y=1$")
axs[3].plot(thetas, poisson_loss(theta=thetas, y=4.0), label=r"$y=4$")
axs[3].plot(thetas, poisson_loss(theta=thetas, y=10.0), label=r"$y=10$")
axs[3].set_xlabel(r"$\theta$")
axs[3].set_title(r"Loss $L(\theta, y)$")
axs[3].legend(loc="upper right")

fig.set_tight_layout(True)

plt.show()
