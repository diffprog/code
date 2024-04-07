# Copyright 2023 Google LLC
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
from sklearn.metrics.pairwise import euclidean_distances


def smoothed_conjugate_conv(f, x, eps=1.0):
  """
  Compute f* via convolution.

  f: array containing the values of f.
  x: grid on which f has been evaluated.
  eps: regularization strength.

  The grid on which f* is evaluated is assumed to be the same.
  """
  x = x.ravel()
  h = np.exp((0.5 * x ** 2 - f) / eps)
  g = np.exp(-0.5 * x ** 2 / eps)
  Kh = np.convolve(g, h, mode='same')
  return eps * np.log(Kh) + 0.5 * x ** 2


def smoothed_conjugate_dot(f, x, y=None, eps=1.0):
  """
  Compute f* via matrix product.

  f: array containing the values of f.
  x: grid on which f has been evaluated.
  y: grid on which to evaluate f*. If None, use x.
  eps: regularization strength.
  """
  if y is None:
    y = x

  h = np.exp((0.5 * x ** 2 - f) / eps)
  D = euclidean_distances(y.reshape(-1, 1), x.reshape(-1, 1), squared=True)
  K = np.exp(-D / (2 * eps))
  Kh = np.dot(K, h)
  return eps * np.log(Kh) + 0.5 * y ** 2


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  smoothed_conjugate = smoothed_conjugate_conv
  #smoothed_conjugate = smoothed_conjugate_dot

  plt.figure()

  x = np.linspace(-3, 3, 500)
  f = x ** 2 + 0.3 * np.sin(6 * np.pi * x)
  plt.plot(x, f, c="k", lw=3)
  conj = smoothed_conjugate(f, x, eps=0.01)
  biconj = smoothed_conjugate(conj, x, eps=0.01)
  plt.plot(x, biconj, lw=3)

  plt.show()
