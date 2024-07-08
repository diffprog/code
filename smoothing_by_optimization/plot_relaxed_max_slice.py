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
from matplotlib import rcParams
from scipy.special import logsumexp as _softmax
import seaborn as sns

rcParams.update({
    'font.size': 16,
    'lines.linewidth': 2,
    'mathtext.fontset': 'dejavusans',
})

def max(u):
  return np.max(u, axis=-1)


def softmax(u):
  return _softmax(u, axis=-1)


def projection_simplex(v, z=1):
  n_features = v.shape[0]
  u = np.sort(v)[::-1]
  cssv = np.cumsum(u) - z
  ind = np.arange(n_features) + 1
  cond = u - cssv / ind > 0
  rho = ind[cond][-1]
  theta = cssv[cond][-1] / float(rho)
  w = np.maximum(v - theta, 0)
  return w


def sparsemax(u):
  z = np.zeros((u.shape[0], u.shape[1]))
  for i in range(u.shape[0]):
    for j in range(u.shape[1]):
      sparseargmax = projection_simplex(u[i, j])
      z[i, j] = u[i, j].dot(sparseargmax) - 0.5*sparseargmax.dot(sparseargmax-1)
  return z


def plot_fn(fn, title):
  u1 = np.linspace(-4, 4, 1000)
  u2 = np.linspace(-4, 4, 1000)
  U1, U2 = np.meshgrid(u1, u2)
  U = np.moveaxis(np.array([U1, U2, np.zeros_like(U1)]), 0, -1)

  fig, ax = plt.subplots(1, 1, figsize=(4, 5.2))

  Z = fn(U)
  im = ax.pcolormesh(U1, U2, Z, cmap=sns.color_palette('rocket', as_cmap=True), linewidth=0, rasterized=True)
  cbar = fig.colorbar(im, ax=ax, orientation='horizontal', label='Value', pad=0.2)
  cbar.ax.locator_params(nbins=3)
  ax.contour(
        U1,
        U2,
        Z,
        10,
        interpolation='none',
        linestyles='dotted',
        alpha=0.8,
        cmap='Greys_r',
    )
  ax.set_xlabel("$u_1$")
  ax.set_ylabel("$u_2$")
  ax.locator_params(axis='y', nbins=3)
  ax.locator_params(axis='x', nbins=3)
  ax.set_title(title, y=1.05)
  fig.tight_layout()
  # fig.savefig(fn.__name__ + '.pdf', format='pdf', bbox_inches='tight')
  plt.show()

plot_fn(max, r'$\mathrm{max}(u_1, u_2, 0)$')
plot_fn(softmax, r'$\mathrm{softmax}(u_1, u_2, 0)$')
plot_fn(sparsemax, r'$\mathrm{sparsemax}(u_1, u_2, 0)$')


