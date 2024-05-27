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
from scipy.special import expit as logistic


def softplus(theta):
  return np.log(1 + np.exp(theta))


def logistic_loss(theta, y):
  return softplus(theta) - y * theta


def plot_vertical_line(ax, x, y, color):
  ax.plot([x, x], [0.0, y], color=color, linestyle='-')
  ax.plot(x, y, color=color, marker="o")


thetas = np.linspace(-5, 5, 100)

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

plot_vertical_line(axs[0], x=0.0, y=0.2, color='C0')
plot_vertical_line(axs[0], x=1.0, y=0.8, color='C0')
axs[0].annotate("0.2", [0.1, 0.2], fontsize=15)
axs[0].annotate("0.8", [0.8, 0.78], fontsize=15)

axs[0].set_xlabel(r"$y$")
axs[0].set_xticks([0, 1])
axs[0].set_title(r"PMF")
axs[0].set_ylim([0, 1.05])

plot_vertical_line(axs[1], x=0.0, y=0.2, color='C0')
plot_vertical_line(axs[1], x=1.0, y=1.0, color='C0')
axs[1].annotate("0.2", [0.1, 0.2], fontsize=15)
axs[1].annotate("1.0", [0.8, 0.98], fontsize=15)

axs[1].set_xlabel(r"$y$")
axs[1].set_xticks([0, 1])
axs[1].set_ylim([0, 1.05])
axs[1].set_title(r"CDF")

axs[2].plot(thetas, logistic(thetas), c="k")
axs[2].set_xlabel(r"$\theta$")
axs[2].set_xticks([-3, 0, 3])
axs[2].set_title(r"Mean function $A'(\theta)$")
axs[2].set_ylim([0, 1.05])

axs[3].plot(thetas, logistic_loss(theta=thetas, y=1), label=r"$y=1$",
            color="C2")
axs[3].plot(thetas, logistic_loss(theta=thetas, y=0), label=r"$y=0$",
            color="C3")
axs[3].set_xlabel(r"$\theta$")
axs[3].set_title(r"Loss $L(\theta, y)$")
axs[3].legend(loc="best")

fig.set_tight_layout(True)

plt.show()
