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

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


rcParams.update({
    'font.size': 16,
    'lines.linewidth': 3,
    'mathtext.fontset': 'dejavusans',
})

sigma = 0.5

def fun(x):
  return np.where(x>=-1., 1, 0)*np.where(x<1, 1, 0)

def sigmoid(x, sigma):
  return 1/(1+np.exp(-x/sigma))

def local_smooth_fun(x, sigma):
  pia = sigmoid(x+1, sigma)
  pib = sigmoid(1-x, sigma)
  return pia*pib

def global_smooth_fun(x, sigma):
  pia = sigmoid(-1-x, sigma)
  pib = sigmoid(1-x, sigma)
  return (pib - pia)

xs = np.linspace(-3, 3, 100)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, sigma in zip(axs, [1.0, 0.5, 0.1]):
  ax.plot(xs, fun(xs), label='Orignal function')
  ax.plot(xs, local_smooth_fun(xs, sigma), label='Locally smoothed')
  ax.plot(xs, global_smooth_fun(xs, sigma), label='Globally smoothed', ls="--")
  ax.set_title(f'$\sigma={sigma}$')
  ax.legend()
  handles, labels = ax.get_legend_handles_labels()
  ax.get_legend().remove()
  ax.locator_params(axis='y', nbins=2)
  ax.locator_params(axis='x', nbins=5)
fig.legend(
    handles,
    labels,
    loc='lower center',
    ncols=3,
    bbox_to_anchor=(0.5, -0.1)
)
fig.tight_layout()
#fig.savefig('global_vs_local_smoothing.pdf', format='pdf', bbox_inches='tight')

plt.show()

# For curiosity, compute total variations
sigma = 2.
local_total_var = np.mean(np.abs(fun(xs) - local_smooth_fun(xs, sigma))**2)
global_total_var = np.mean(np.abs(fun(xs) - global_smooth_fun(xs, sigma))**2)
print(f'local total var: {local_total_var}, global total var: {global_total_var}')
