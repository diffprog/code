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
import seaborn as sns


def r(i, j, D, N):
  if j % 2 == 0:
    denom = N ** (j / D)
    return np.sin(i / denom)
  else:
    denom = N ** ((j-1) / D)
    return np.cos(i / denom)


def positional_encoding_matrix(L, D, N):
  P = np.zeros((L, D))
  for i in range(L):
    for j in range(D):
      P[i, j] = r(i, j, D, N)
  return P


def plot_wave(ax, j, L, D, N, yticks=False):
  i = np.arange(0, L, 1)
  values = r(i, j, D, N)
  ax.plot(values, i, c="C%d" % j)
  ax.set_xlabel("j = %d" % j, size=16)
  if not yticks:
    ax.set_yticks([])


def plot_waves(L, D, N):
  fig, axes = plt.subplots(1, 6, figsize=(4 * 6,4))
  plot_wave(axes[5], j=0, L=L, D=D, N=N)
  plot_wave(axes[4], j=1, L=L, D=D, N=N)
  plot_wave(axes[3], j=2, L=L, D=D, N=N)
  plot_wave(axes[2], j=3, L=L, D=D, N=N)
  plot_wave(axes[1], j=4, L=L, D=D, N=N)
  plot_wave(axes[0], j=5, L=L, D=D, N=N, yticks=True)
  axes[0].set_ylabel(r"Position i", size=16)
  axes[0].set_xlabel(r"Dimension j = 5", size=16)


def plot_heatmap(L, D, N):
  plt.figure()
  P = positional_encoding_matrix(L, D, N)
  sns.heatmap(P)
  plt.ylabel("Position i", size=16)
  plt.xlabel("Dimension j", size=16)


plot_waves(L=200, D=6, N=30)

plot_heatmap(L=100, D=50, N=30)

plt.show()
