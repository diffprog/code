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
import functools

from matplotlib import pyplot as plt
from matplotlib import rcParams

import numpy as np
import seaborn as sns

rcParams.update({
    'font.size': 16,
    'lines.linewidth': 2,
    'mathtext.fontset': 'dejavusans',
})

# Objective function
a, b = 0.1, 1.
ab = np.array([a, b])
def contour_fun(x, y):
  return 0.5*a*x**2 + 0.5*b*y**2

def fun(w):
  return 0.5*np.sum(ab*w**2)

def grad_fun(w):
  return ab*w

# Optimization loop
def run_gd(stepsize, momentum, maxiter):
  w = np.array([-0.75, 0.25])
  cumul_grad = np.zeros_like(w)
  ws = [w]
  values = [fun(w)]
  for _ in range(maxiter):
    cumul_grad = momentum*cumul_grad - stepsize*grad_fun(w)
    w = w + cumul_grad
    ws.append(w)
    values.append(fun(w))
  return ws, values

def plot_optimization_steps(stepsize, momentum, maxiter):
  ws, values = run_gd(stepsize, momentum, maxiter)

  # Plot contour at values of the iterates
  fig, ax = plt.subplots(1, 1, figsize=(8, 4.6))
  x = np.linspace(-1, 1., 100)
  y = np.linspace(-1., 1., 100)
  X, Y = np.meshgrid(x, y)
  Z = contour_fun(X, Y)
  Z = np.ma.masked_where(Z > 0.08, Z)
  ax.contour(
      X,
      Y,
      Z,
      np.array(values[::-1]),
      interpolation='none',
      linestyles='dotted',
      cmap=sns.color_palette('flare', as_cmap=True)
  )

  # Plot iterates
  margin = (0.02, 0.015)
  for i, w in enumerate(ws[:3]):
    ax.plot(w[0], w[1], 'x', color='k')
    ax.annotate(f'$w^{(i)}$', xy=(w[0], w[1]), xytext=(w[0] + margin[0], w[1] + margin[1]))

  # Plot arrows between points
  for w1, w2 in zip(ws[:-1], ws[1:]):
    ax.annotate(
        '',
        xytext=(w1[0], w1[1]),
        xy=(w2[0], w2[1]),
        arrowprops=dict(color='k', width=0.5, headwidth=4, alpha=0.5),
    )

  # Plot optimum
  ax.plot(0, 0, '+', color='tab:red')
  ax.annotate('$w^*$', xy=(0, 0), xytext=(0.02, -0.01), color='tab:red')

  # Add labels and zoom
  ax.set_xlabel('$w_1$')
  ax.set_ylabel('$w_2$')
  ax.set_ylim(-0.5, 0.5)
  ax.locator_params(axis='y', nbins=3)
  ax.locator_params(axis='x', nbins=3)

  # Add title
  title = 'Gradient descent' if momentum == 0. else 'Gradient descent with momentum'
  title = title + f'\n Stepsize {stepsize}'
  if momentum > 0.:
    title = title + f' Momentum {momentum}'
  ax.set_title(title)
  fig.tight_layout()
  # fig.savefig(
  #   'grad_descent_step_{0}_mom_{1}.pdf'.format(
  #     str(stepsize).replace('.', '_'), str(momentum).replace('.', '_')
  #   ),
  #   format='pdf',
  #   bbox_inches='tight'
  # )

plot_optimization_steps(stepsize=0.5, momentum=0., maxiter=5)
plot_optimization_steps(stepsize=1.8, momentum=0., maxiter=5)
plot_optimization_steps(stepsize=1.8, momentum=0.2, maxiter=5)
plt.show()
