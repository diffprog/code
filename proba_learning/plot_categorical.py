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
from scipy.special import softmax as softargmax
from scipy.special import logsumexp as softmax


def logistic_loss(theta, y):
  return softmax(theta) - np.dot(theta, y)


def plot_vertical_line(ax, x, y, color):
  ax.plot([x, x], [0.0, y], color=color, linestyle='-')
  ax.plot(x, y, color=color, marker="o")


s_values = np.linspace(-5, 5, 100)

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

plot_vertical_line(axs[0], x=1, y=0.3, color='C0')
plot_vertical_line(axs[0], x=2, y=0.6, color='C0')
plot_vertical_line(axs[0], x=3, y=0.1, color='C0')
axs[0].annotate("0.3", [0.3, 0.3], fontsize=15)
axs[0].annotate("0.6", [1.3, 0.6], fontsize=15)
axs[0].annotate("0.1", [2.3, 0.1], fontsize=15)

axs[0].set_xlabel(r"$y$")
axs[0].set_xlim([0, 4])
axs[0].set_xticks([1, 2, 3])
axs[0].set_title(r"PMF")
axs[0].set_ylim([0, 1.05])

plot_vertical_line(axs[1], x=1, y=0.3, color='C0')
plot_vertical_line(axs[1], x=2, y=0.9, color='C0')
plot_vertical_line(axs[1], x=3, y=1.0, color='C0')
axs[1].annotate("0.3", [0.3, 0.25], fontsize=15)
axs[1].annotate("0.9", [1.3, 0.85], fontsize=15)
axs[1].annotate("1.0", [2.3, 0.95], fontsize=15)

axs[1].set_xlabel(r"$y$")
axs[1].set_xlim([0, 4])
axs[1].set_xticks([1, 2, 3])
axs[1].set_ylim([0, 1.05])
axs[1].set_title(r"CDF")

axs[2].plot(s_values, [softargmax([s, 1, 0])[0] for s in s_values], c="C2")
axs[2].plot(s_values, [softargmax([s, 1, 0])[1] for s in s_values], c="C3")
axs[2].plot(s_values, [softargmax([s, 1, 0])[2] for s in s_values], c="C4",
            ls="--")
axs[2].set_xlabel(r"$s$")
axs[2].set_xticks([-3, 0, 3])
axs[2].set_title(r"$\langle \nabla A(\theta), y \rangle$")
axs[2].set_ylim([0, 1.05])

axs[3].plot(s_values, [logistic_loss(theta=[s, 1, 0], y=[1, 0, 0])
                       for s in s_values], label=r"$y=e_1$", color="C2")
axs[3].plot(s_values, [logistic_loss(theta=[s, 1, 0], y=[0, 1, 0])
                       for s in s_values], label=r"$y=e_2$", color="C3")
axs[3].plot(s_values, [logistic_loss(theta=[s, 1, 0], y=[0, 0, 1])
                       for s in s_values], label=r"$y=e_3$", color="C4", ls="--")
axs[3].set_xlabel(r"$s$")
axs[3].set_title(r"Loss $L(\theta, y)$")
axs[3].legend(loc="best")

fig.set_tight_layout(True)

plt.show()
