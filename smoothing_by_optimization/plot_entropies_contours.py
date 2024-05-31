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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker, cm
from functools import partial


_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])
_midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3]) / 2.0
              for i in range(3)]


def xy2bc(xy, tol=1.e-3):
  s = [(_corners[i] - _midpoints[i]).dot(xy - _midpoints[i]) / 0.75
       for i in range(3)]

  return np.clip(s, tol, 1.0 - tol)


def bc2xy(P):
  return np.dot(P, _corners)


def draw_contours(f, nlevels=20, subdiv=8, **kwargs):
    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [f(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    return plt.tricontour(trimesh, pvals, nlevels, **kwargs)


def draw(f, **kwargs):
  # draw the triangle
  plt.triplot(_triangle, color='k')

  # label the corners
  for i, corner in enumerate(_corners):
    p1, p2, p3 = xy2bc(corner).round(1)
    label = "({:1.0f}, {:1.0f}, {:1.0f})".format(p1, p2, p3)
    va = 'top'
    sgn = -1
    if p3 == 1:
      va = 'bottom'
      sgn = +1.2

    plt.annotate(label, xy=corner,
                 xytext=(0, sgn * 5),
                 textcoords='offset points',
                 horizontalalignment='center',
                 verticalalignment=va)

  cs = draw_contours(f, subdiv=8,
                     cmap=plt.cm.plasma_r,
                     **kwargs)
  plt.xlim(-0.1, 1.1)
  plt.ylim(-0.1, _corners[-1][-1] + 0.1)
  plt.axis('off')
  return cs


def shannon_entropy(p):
  p = np.array(p)
  mask = p > 0
  plogp = np.zeros_like(p)
  plogp[mask] = p[mask] * np.log(p[mask])
  return -np.sum(plogp)


def tsallis_entropy(p, alpha=1.5):
  p = np.array(p)
  scale = 1./ (alpha * (alpha - 1))
  return scale * (1 - np.sum(p ** alpha))


plt.figure(figsize=(12, 3))

mpl.rc('lines', linewidth=1.5)
mpl.rc('font', size=16)

plt.subplot(131)
draw(shannon_entropy)
plt.title(r"Tsallis $\alpha \to 1$ (Shannon)", y=1.08)
plt.axis("off")

plt.subplot(132)
plt.title(r"Tsallis $\alpha=1.5$", y=1.08)
draw(partial(tsallis_entropy, alpha=1.5))
plt.axis("off")

plt.subplot(133)
plt.title(r"Tsallis $\alpha=2$ (Gini)", y=1.08)
draw(partial(tsallis_entropy, alpha=1.5))
plt.axis("off")

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.96, top=0.80,
                    wspace=0.32, hspace=0.2)

plt.show()
