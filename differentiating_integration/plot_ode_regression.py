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
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.collections as mcoll
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

rcParams.update({
    'font.size': 16,
    'lines.linewidth': 3,
    'mathtext.fontset': 'dejavusans',
})


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
  new_cmap = colors.LinearSegmentedColormap.from_list(
      'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
      cmap(np.linspace(minval, maxval, n)),
  )
  return new_cmap


def colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap('copper'),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=None,
    alpha=1.0,
    segment_interval=0,
    label=None,
    ax=None,
):
  """Plot a colored line with coordinates x and y.

  Adapted from
  https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line
  """

  # Default colors equally spaced on [0,1]:
  if z is None:
    z = np.linspace(0.0, 1.0, len(x))

  z = np.asarray(z)
  k = segment_interval
  for i in range(int(k / 2)):
    z[i::k] = 1.0

  points = np.array([x, y]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)

  lc = mcoll.LineCollection(
      segments,
      array=z,
      cmap=cmap,
      norm=norm,
      linestyles='solid',
      linewidth=linewidth,
      alpha=alpha,
      label=label,
  )
  if ax is None:
    ax = plt.gca()
  ax.add_collection(lc)

  return lc


def get_mean_colormap(cmap):
  ncolors = cmap.N
  indices = np.linspace(0, ncolors - 1, ncolors)
  colors = cmap(indices)
  return np.mean(colors, axis=0)


def cont_dyn(z, hparams):
  a, b, c, d = hparams
  xdot = a * z[0] - b * z[0] * z[1] + 0.01
  ydot = d * z[0] * z[1] - c * z[1] + 0.01
  return np.array([xdot, ydot])


def run_ode(z0, hparams, dt, T):
  maxiter = int(T / dt)
  z = z0
  zs = []
  for _ in range(maxiter):
    z = z + dt * cont_dyn(z, hparams)
    zs.append(z)
  return np.array(zs)


fig, ax = plt.subplots(1, 1)

true_hparams = (1 / 3, 4 / 3, 1, 1)

rng = np.random.default_rng(0)
z0 = np.ones(2)
dt = 1e-3
T = 30
zs = run_ode(z0, true_hparams, dt, T)

n = 100
eps = 1e-2
select_idxs = ((T / dt) / n * np.arange(n)).astype(int)
z_train = zs[select_idxs] + eps * rng.normal(size=(n, 2))

ax.plot(z0[0], z0[1], marker='x', color='k')
ax.scatter(
    z_train[:, 0],
    z_train[:, 1],
    c=np.arange(n),
    cmap=sns.color_palette('rocket', as_cmap=True),
)

hparams1 = (1.5 / 3, 4 / 3, 1, 1)
hparams2 = (0.5 / 3, 4 / 3, 1.0, 1)

palettes = [
    matplotlib.colormaps['Greens_r'],
    matplotlib.colormaps['Oranges_r'],
    matplotlib.colormaps['Blues_r'],
]
handles, labels = [], []
for i, hparams in enumerate([hparams1, hparams2, true_hparams]):
  zs = run_ode(z0, hparams, dt, T)
  if hparams != true_hparams:
    colorline(
        zs[:, 0],
        zs[:, 1],
        alpha=0.6,
        cmap=palettes[i],
        segment_interval=200,
        ax=ax,
    )
    handles.append(
        Line2D(
            [0],
            [0],
            label=f'$s(t; w_{i+1})$',
            linestyle='dashed',
            color=get_mean_colormap(truncate_colormap(palettes[i], maxval=0.5)),
        )
    )
  else:
    colorline(
        zs[:, 0],
        zs[:, 1],
        cmap=truncate_colormap(palettes[i], maxval=0.5),
        ax=ax,
    )
    handles.append(
        Line2D(
            [0],
            [0],
            label=rf'$s(t; w_*)$',
            color=get_mean_colormap(truncate_colormap(palettes[i], maxval=0.5)),
        )
    )

plt.legend(handles=handles)
ax.set_xlim(0.15, 1.85)
ax.locator_params(axis='y', nbins=3)
ax.locator_params(axis='x', nbins=3)
fig.tight_layout()
# fig.savefig('ode_regression.pdf', format='pdf', bbox_inches='tight')

plt.show()
