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

from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy.special import softmax as _softargmax
import seaborn as sns

rcParams.update({
    'font.size': 14,
    'lines.linewidth': 2,
    'mathtext.fontset': 'dejavusans',
})

palette = sns.color_palette('rocket', as_cmap=True)

# Setup simplex
# Scale controls the scale of the xyz coordinates
scale = 4
corners = np.array([[0, 0], [scale, 0], [scale / 2, scale * 0.75**0.5]])
center = np.mean(corners, axis=0)
midpoints = np.array(
    [(corners[i % 3] + corners[(i + 1) % 3]) / 2.0 for i in range(3)]
)

dist_center_to_edge = scale * np.sqrt(3 / 4) / 3

axes = corners - center

middle_axes = midpoints - center
middle_axes_norms = np.expand_dims(np.linalg.norm(middle_axes, axis=1), axis=-1)
middle_axes = 1 / middle_axes_norms * middle_axes


def plot_fn(fn, title):
  fig, axs = plt.subplots(1, 3, figsize=(3 * 4, 5))
  a = np.linspace(0, scale, 1000)
  b = np.linspace(0, scale, 1000)
  A, B = np.meshgrid(a, b)
  AB = np.moveaxis(np.array([A, B]), 0, -1)
  AB_to_middle_axes = (AB - center).dot(middle_axes.T)
  xyz = (AB - center).dot(axes.T)
  for coord, ax in enumerate(axs):
    # Heatmap
    C = np.where(
        np.max(AB_to_middle_axes, axis=2) >= dist_center_to_edge,
        np.nan,
        fn(xyz, coord),
    )
    C = np.ma.masked_where(np.isnan(C), C)
    im = ax.pcolormesh(
        A, B, C, cmap=palette, linewidth=0, rasterized=True, vmin=0, vmax=1
    )
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', label='Value')
    cbar.ax.locator_params(nbins=3)

    # Contour plot
    # Restrict slightly the mask to avoid the contour to touch the triangle edges
    C2 = np.where(
        np.max(AB_to_middle_axes, axis=2) >= dist_center_to_edge - 1e-2,
        np.nan,
        fn(xyz, coord),
    )
    if fn != argmax:
      ax.contour(
          A,
          B,
          C2,
          10,
          interpolation='none',
          linestyles='dotted',
          alpha=0.8,
          cmap='Greys_r',
      )

    # Add x, y, z axes
    margins = (
        np.array([scale * 0.12, -scale * 0.01]),
        np.array([0.0, scale * 0.05]),
        np.array([scale * 0.03, 0.0]),
    )
    for i, margin in enumerate(margins):
      end_point = corners[i] + 0.3 * axes[i]
      end_point2 = corners[i] + 0.32 * axes[i]
      ax.plot(
          (center[0], end_point[0]),
          (center[1], end_point[1]),
          color='k',
          zorder=100,
          linewidth=0.5,
      )
      ax.annotate(
          '',
          xytext=(center[0], center[1]),
          xy=(end_point2[0], end_point2[1]),
          arrowprops=dict(color='k', width=0.5, headwidth=8),
      )
      ax.annotate(f'$u_{i+1}$', end_point, xytext=end_point + margin)
      ax.annotate(f'$u_{i+1}$', end_point, xytext=end_point + margin)

    if fn != argmax:
      ax.annotate('0', center, xytext=(center[0] - scale * 0.1, center[1]))

    ax.set_title(f'$\pi_{coord+1}$', y=1.1)
    ax.axis('off')
  fig.suptitle(title, y=0.95)
  fig.tight_layout()
  fig.savefig(fn.__name__ + '.pdf', format='pdf', bbox_inches='tight')


def argmax(u, coord):
  return np.where(np.argmax(u, axis=-1) == coord, 1, 0)


def softargmax(u, coord):
  return _softargmax(u, axis=-1)[..., coord]


plot_fn(argmax, r'$\boldsymbol{\pi} = \mathrm{argmax}(\boldsymbol{u})$')
plot_fn(softargmax, r'$\boldsymbol{\pi} = \mathrm{softargmax}(\boldsymbol{u})$')

plt.show()
