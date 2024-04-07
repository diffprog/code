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

"""
Smoothing out the soft thresholding operator.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as logistic
from scipy.special import softmax as softargmax


def st(x, lam):
  if x >= lam:
    return x - lam
  elif x <= -lam:
    return x + lam
  else:
    return 0


def argmax(x):
  i = np.argmax(x)
  e = np.zeros_like(x)
  e[i] = 1
  return e


def step(u):
  if u >= 0:
    return 1
  else:
    return 0


def _mean_std(x, lam, pi):
  # Three possible output values.
  g = np.array([0, x - lam, x + lam])
  # The mean is the convex combination of the possible output values.
  m = np.dot(pi, g)
  # We also compute the variance.
  v = np.mean((m - g) ** 2)
  # Return mean and std deviation.
  return m, np.sqrt(v)


# As explained in the book, the vector `pi` containing the probabilities of each
# branch can either be defined by combining logical operators or by using an
# argmax. In general, the two approaches do not exactly lead to the same
# probability distribution.

def st_from_step(x, lam, step_fn=step):
  pi = np.array([step_fn(x + lam) * step_fn(lam - x),
                 step_fn(x - lam),
                 step_fn(-x - lam)])
  return _mean_std(x, lam, pi)


def st_from_argmax(x, lam, argmax_fn=argmax):
  a = np.array([lam - np.abs(x), x - lam, -x - lam])
  pi = argmax_fn(a)
  return _mean_std(x, lam, pi)


values = np.linspace(-5, 5, 100)
lam = 1.0

plt.figure()

if len(sys.argv) == 1:
  # If no option is provided, we use the argmax-based formulation.
  m, std = zip(*[st_from_argmax(v, lam, argmax_fn=softargmax) for v in values])
else:
  # If any option is provided, we use the step-based formulation.
  m, std = zip(*[st_from_step(v, lam, step_fn=logistic) for v in values])

m = np.array(m)
std = np.array(std)

plt.plot(values, m, lw=4, label="Mean")
plt.fill_between(values, m - std, m + std, facecolor='yellow', alpha=0.3,
                 label="Standard deviation")

plt.plot(values, [st(v, lam) for v in values], c="k", label="Hard", ls="--")

plt.legend(loc="upper left", fontsize=18)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.axhline(0, c="k", alpha=0.3)
plt.axvline(0, c="k", alpha=0.3)

plt.show()
