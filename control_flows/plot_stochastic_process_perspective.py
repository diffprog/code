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


def gaussian_pdf(x, mu, sigma):
  pre_factor = 1 / (sigma * np.sqrt(2 * np.pi))
  exponent = -((x - mu)**2) / (2 * sigma**2)
  return pre_factor * np.exp(exponent)


xs = np.linspace(-3, 15, 100)

plt.rcParams.update(
    {'lines.linewidth': 4,
    'legend.fontsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.titlesize': 24,
    })

mu1 = 3
mu2 = 9

plt.figure()


plt.plot(xs, gaussian_pdf(xs, mu=mu1, sigma=1.0))
plt.plot(xs, gaussian_pdf(xs, mu=mu2, sigma=1.0))

plt.annotate(r'$\mu_1 = %d$' % mu1, xy=(mu1 - 1.0, 0.42), fontsize=16)
plt.annotate(r'$\mu_2 = %d$' % mu2, xy=(mu2 - 1.0, 0.42), fontsize=16)

plt.annotate(r'$U_1$', xy=(mu1 - 0.5, 0.1), fontsize=16)
plt.annotate(r'$U_2$', xy=(mu2 - 0.5, 0.1), fontsize=16)

plt.ylim((0, 0.5))

plt.show()
