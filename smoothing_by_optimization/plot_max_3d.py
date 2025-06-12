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
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

rcParams.update({
    'font.size': 16,
    'lines.linewidth': 2,
    'mathtext.fontset': 'dejavusans',
})

x = np.arange(0, 5.1, 0.2)
y = np.arange(0, 5.1, 0.2)
X, Y = np.meshgrid(x, y)

Z_lse = np.log(np.exp(X) + np.exp(Y))
Z_max = np.maximum(X, Y)

fig = plt.figure(figsize=(14, 6))
cmap = sns.color_palette('rocket', as_cmap=True)

ax1 = fig.add_subplot(121, projection='3d')
surf2 = ax1.plot_surface(X, Y, Z_max, cmap=cmap, edgecolor='black',
                         linewidth=0.5, rstride=2, cstride=2)
ax1.set_zlim(0, np.max(Z_max) + 1)
cset = ax1.contour(X, Y, Z_max, zdir='z', offset=0, cmap=cmap)

ax1.view_init(elev=20, azim=-170)
ax1.set_title(r'$\max$', y=0.98)
ax1.set_xlabel(r'$u_2$')
ax1.set_ylabel(r'$u_1$')
ax1.set_zlabel(r'$\mathrm{max}(u_1, u_2)$')

ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

ax1.zaxis.set_rotate_label(False)
ax1.zaxis.label.set_rotation(92)

ax2 = fig.add_subplot(122, projection='3d')
surf1 = ax2.plot_surface(X, Y, Z_lse, cmap=cmap, edgecolor='black',
                         linewidth=0.5, rstride=2, cstride=2)

ax2.set_zlim(0, np.max(Z_lse) + 1)
cset = ax2.contour(X, Y, Z_lse, zdir='z', offset=0, cmap=cmap)

ax2.view_init(elev=20, azim=-170)
ax2.set_title(r'$\mathrm{softmax}$', y=0.98)
ax2.set_xlabel(r'$u_2$')
ax2.set_ylabel(r'$u_1$')
ax2.set_zlabel(r'$\mathrm{softmax}(u_1, u_2)$')

ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

ax2.zaxis.set_rotate_label(False)
ax2.zaxis.label.set_rotation(92)

plt.subplots_adjust(wspace=-0.1)
plt.show()

#fig.savefig('max_softmax_3d' + '.pdf', format='pdf', bbox_inches='tight')
