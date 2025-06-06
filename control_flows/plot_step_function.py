# Copyright 2025 Google LLC
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


def heaviside(u):
    return np.where(u > 0, 1, 0)


def logistic(u):
  return 1. / (1 + np.exp(-u))


xs = np.linspace(-3, 3, 100)

plt.rcParams.update(
    {'lines.linewidth': 4,
    'legend.fontsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.titlesize': 24,
    })

plt.figure()
plt.plot(xs, heaviside(xs), label="Binary")
plt.plot(xs, logistic(xs), label="Logistic")
plt.title("Binary step and logistic functions")
plt.legend(loc="best")
plt.show()
