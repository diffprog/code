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
from matplotlib import pyplot as plt
from scipy.special import expit as logistic


def forward_diff(fun, x):
  def forward_diff_fun(delta):
    return (fun(x + delta) - fun(x)) / delta
  return forward_diff_fun


def central_diff(fun, x):
  def central_diff_fun(delta):
    return (fun(x + delta) - fun(x - delta)) / (2 * delta)
  return central_diff_fun


def complex_diff(fun, x):
  def complex_diff_fun(delta):
    x_eval = x + 1j * delta
    return fun(x_eval).imag / delta
  return complex_diff_fun


# Function at hand, point, derivative
def softplus(x):
  return np.log1p(np.exp(x))


x = 1.
true_derivative = logistic(x)

# Instantiations
forward_diff_fun = forward_diff(softplus, x)
central_diff_fun = central_diff(softplus, x)
complex_diff_fun = complex_diff(softplus, x)

# Evaluations
deltas = np.logspace(-14, 0, 100)

forward_diff_evals = forward_diff_fun(deltas)
central_diff_evals = central_diff_fun(deltas)
complex_diff_evals = complex_diff_fun(deltas)

approx_forward_diffs = np.abs(true_derivative - forward_diff_evals)
approx_central_diffs = np.abs(true_derivative - central_diff_evals)
approx_complex_diffs = np.abs(true_derivative - complex_diff_evals)
approx_complex_diffs = np.maximum(approx_complex_diffs, 1e-17)

# Plots
fig = plt.figure()
#plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'xtick.labelsize' : 14, 'ytick.labelsize': 14})

plt.plot(deltas, approx_forward_diffs)
plt.plot(deltas, approx_central_diffs)
plt.plot(deltas, approx_complex_diffs)

plt.vlines(x = 10**(-7.5), ymin=10**(-10), ymax=10**(-1), color='k',
           linestyle='--')
plt.annotate(xy=(10**(-11), 10**(-3)), text='Round-off error \n is dominant',
             ha='center', fontsize=15)
plt.annotate(xy=(10**(-4), 10**(-3)), text='Truncation error \n is dominant',
             ha='center', fontsize=15)

plt.legend(['Forward difference', 'Central difference', 'Complex Step'],
           fontsize=14, loc='lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\delta$', size=16)
plt.ylabel('Approximation error', size=16)
plt.tight_layout()
plt.show()




