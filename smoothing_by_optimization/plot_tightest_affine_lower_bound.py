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
import seaborn as sns

rcParams.update({
    'font.size': 14,
    'lines.linewidth': 4,
    'mathtext.fontset': 'dejavusans',
})


fig, ax = plt.subplots(1, 1, figsize=(6, 4))

num_lines = 4
palette = sns.color_palette('plasma', num_lines)


def fun(x):
  return x * np.log(x) + (1 - x) * np.log(1 - x)


def conj_fun(u):
  return np.log(1 + np.exp(u))


###############
# Envelope plot
xs = np.linspace(0.0, 1.0, 100)
conj_domain = (-2.0, 1.0)

# Plot function
ys = fun(xs)
ax.plot(xs, ys, color='k')

# Plot tangents
u = 0.7
for i, eps in enumerate(np.linspace(0, 0.5, num_lines)):
  ys = u * xs - conj_fun(u) - eps
  # Plot affine
  ax.plot(xs, ys, '--', color=palette[i], linewidth=2)
  # Plot intercept
  if i == 0:
    ax.plot(0.0, -conj_fun(u), 'o', color=palette[i])
    # Add legend to intercept
    ax.annotate(
        '$-f^*(v)$',
        (0, -conj_fun(u)),
        xytext=(-0.2, -conj_fun(u) - 0.12),
        color=palette[i],
        fontsize=16,
    )
    # Add legend to slope (indicating what v corresponds to)
    ax.plot(
        xs[:20],
        np.zeros_like(xs[:20]) - conj_fun(u),
        '--',
        linewidth=1,
        color=palette[i],
    )
    ax.vlines(
        xs[19],
        -conj_fun(u),
        u * xs[19] - conj_fun(u),
        linestyles='--',
        linewidth=1,
        color=palette[i],
    )
    ax.annotate(
        '$v$',
        (xs[19], u * xs[8] - conj_fun(u)),
        xytext=(xs[19] + 0.03, u * xs[8] - conj_fun(u)),
        color=palette[i],
    )

# Approriate zoom in y axis
ax.set_ylim(-1.4, 0.0)

# Set y_axis to the left
ax.spines['left'].set_position(('data', 0))

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Less ticks
ax.locator_params(axis='y', nbins=2)
ax.locator_params(axis='x', nbins=2)

# Labels
ax.set_xlabel('$u$', fontsize=16)
ax.set_ylabel('$f\,(u)$', fontsize=16)
# ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

fig.tight_layout()
# fig.savefig('tightest_affine_lower_bound.pdf', format='pdf', bbox_inches='tight')
plt.show()
