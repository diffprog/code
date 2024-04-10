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
import matplotlib.pylab as plt


def plot_convergence_rates():
  t = np.arange(1, 100)

  fig = plt.figure()
  plt.plot(t, 1. / np.sqrt(t), label=r"$R(t) = 1/\sqrt{t}$ (sublinear)", lw=4)
  plt.plot(t, 1. / t, label=r"$R(t) = 1/t$ (sublinear)", lw=4)
  plt.plot(t, 1. / (t ** 2), label=r"$R(t) = 1/t^2$ (sublinear)", lw=4)
  plt.plot(t, np.exp(-t), label="$R(t) = e^{-t}$ (linear)", lw=4)
  plt.plot(t, np.exp(-t ** 2), label="$R(t) = e^{-t^2}$ (superlinear)", lw=4)
  plt.ylim((1e-20, 1))
  plt.yscale("log")
  plt.ylabel(r"Convergence rate $R(t)$ (log scale)", fontsize=17)
  plt.xlabel(r"Iteration $t$", fontsize=17)
  plt.legend(loc="best", fontsize=17)
  fig.set_tight_layout(True)


def ratios(x, q=1.0):
  ret = []
  for i in range(1, len(x)):
    ret.append(x[i] / (x[i-1] ** q))
  return np.array(ret)


def plot_progress_ratios():
  t = np.arange(1, 100)

  fig = plt.figure()
  plt.plot(t[1:], ratios(1. / np.sqrt(t)),
           label=r"$R(t) = 1/\sqrt{t}$ (sublinear)", lw=4)
  plt.plot(t[1:], ratios(1. / t), label=r"$R(t) = 1/t$ (sublinear)", lw=4)
  plt.plot(t[1:], ratios(1. / (t ** 2)),
           label=r"$R(t) = 1/t^2$ (sublinear)", lw=4)
  plt.plot(t[1:], ratios(np.exp(-t)), label="$R(t) = e^{-t}$ (linear)", lw=4)
  plt.plot(t[1:], ratios(np.exp(-t ** 2)),
           label="$R(t) = e^{-t^2}$ (superlinear)", lw=4)
  plt.ylim((1e-20, 1))
  plt.ylabel(r"Progress ratio $\rho_t = \frac{R(t)}{R(t-1)}$", fontsize=17)
  plt.xlabel(r"Iteration $t$", fontsize=17)
  #plt.legend(loc="best", fontsize=17)
  fig.set_tight_layout(True)


if __name__ == '__main__':
  plot_convergence_rates()
  plot_progress_ratios()
  plt.show()
