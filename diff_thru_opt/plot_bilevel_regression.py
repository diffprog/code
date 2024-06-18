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
    'lines.linewidth': 3,
    'mathtext.fontset': 'dejavusans',
})

n = 100
rng = np.random.default_rng(2)
eps = 1.0
true_sigma = 0.2
palette = sns.color_palette('rocket', 3)


def kernel(a, b, sigma):
  return np.exp(-0.5 * (a - b) ** 2 / sigma**2)


def make_predictor(x_train, y_train, sigma):
  X, XT = np.meshgrid(x_train, x_train)
  K_train = kernel(XT, X, sigma) + 0.001 * np.eye(len(x_train))
  coefs = np.linalg.solve(K_train, y_train)

  def predict(x):
    k = kernel(x_train, x, sigma)
    return coefs.dot(k)

  return predict


def make_data(n, true_sigma):
  x = 2 * rng.uniform(size=n)
  X, XT = np.meshgrid(x, x)
  K = kernel(XT, X, sigma=true_sigma)
  y = K.dot(rng.normal(size=n))
  noise = rng.normal(size=n)
  return x, y, noise


def plot_predictor(x, y, noise, sigma, frac_train, eps, ax, idx):
  x_train = x[: int(frac_train * n)]
  y_train = y[: int(frac_train * n)] + eps * noise[: int(frac_train * n)]
  x_val = x[int(frac_train * n) :]
  y_val = y[int(frac_train * n) :]
  # Plot data
  ax.scatter(x_train, y_train, color='tab:blue')
  ax.scatter(x_val, y_val, color='tab:cyan', marker='D')

  # Plot predictor
  predict = make_predictor(x_train, y_train, sigma)
  xs = np.linspace(np.min(x), np.max(x), 100)
  ys = []
  for x in xs:
    ys.append(predict(x))
  ys = np.array(ys)
  if sigma != true_sigma:
    ax.plot(
        xs,
        ys,
        label=f'$w^*(\lambda_{idx+1})$',
        color=palette[idx],
        alpha=0.6,
        linestyle='dashed',
    )
  else:
    ax.plot(xs, ys, label=f'$w^*(\lambda^*)$', color=palette[idx])


x, y, noise = make_data(n, true_sigma)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
legends = []
for idx, sigma in enumerate([0.1 * true_sigma, 10 * true_sigma, true_sigma]):
  plot_predictor(x, y, noise, sigma, frac_train=2 / 3, eps=1.0, ax=ax, idx=idx)
  ax.locator_params(axis='y', nbins=3)
  ax.locator_params(axis='x', nbins=3)
  ax.set_ylim(-10, 5)

ax.legend()
fig.tight_layout()
# fig.savefig('bilevel_illustration.pdf', format='pdf')
plt.show()
