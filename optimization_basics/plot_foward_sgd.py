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
    'font.size': 16,
    'lines.linewidth': 2,
    'mathtext.fontset': 'dejavusans',
})

a, b = 0.1, 1.
ab = np.array([a, b])
def contour_fun(x, y):
  return 0.5*a*x**2 + 0.5*b*y**2

def fun(w):
  return 0.5*np.sum(ab*w**2)

def grad_fun(w):
  return ab*w

def run_gd(initial_point, stepsize, maxiter):
  w = initial_point
  ws = [w]
  values = [fun(w)]
  for _ in range(maxiter):
    w = w - stepsize*grad_fun(w)
    ws.append(w)
    values.append(fun(w))
  return np.array(ws)

def run_forward_gd(initial_point, stepsize, maxiter):
  w = initial_point
  ws = [w]
  values = [fun(w)]
  for _ in range(maxiter):
    direction = np.eye(2)[np.random.randint(0, 2)]
    forward_grad = grad_fun(w).dot(direction) * direction
    w = w - stepsize*forward_grad
    ws.append(w)
    values.append(fun(w))
  return np.array(ws)

stepsize = 1.6
maxiter = 20

initial_point = np.array([-7, 3.8])
gd_path = run_gd(initial_point, stepsize, maxiter)
forward_gd_path = run_forward_gd(initial_point, stepsize, maxiter)

x = np.linspace(-8, 8, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)
Z = contour_fun(X, Y)

fig = plt.figure(figsize=(8, 4))

num_levels = 8
levels = np.logspace(-2, 0.5, num_levels, base=10)
linewidths = np.linspace(0.5, 3.0, num_levels)
plt.contour(X, Y, Z, levels=levels, cmap=sns.color_palette('flare', as_cmap=True), linewidths=linewidths)

plt.plot(gd_path[:, 0], gd_path[:, 1], '-o', label='Reverse mode')
plt.plot(forward_gd_path[:, 0], forward_gd_path[:, 1], '-o', label='Randomized forward mode')
plt.plot(initial_point[0], initial_point[1], 'ko', markersize=10)

plt.legend(loc="lower right")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.box(False)

plt.show()

fig.savefig('gd_vs_fgd.pdf', format='pdf', bbox_inches='tight')
