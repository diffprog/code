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
import matplotlib.pyplot as plt


def gaussian_kernel(left, right, n_samples, sigma):
  x = np.linspace(left, right, n_samples)
  kernel = np.exp(-0.5 * (x/sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
  # Renormalize the kernel, as we only drew a finite number of samples.
  kernel /= np.sum(kernel)
  return x, kernel


def smooth_signal(f, sigma):
  _, kernel = gaussian_kernel(left=-5, right=5, n_samples=1000, sigma=sigma)
  return np.convolve(f, kernel, mode="same")


n_samples = 1000
x = np.linspace(-4, 4, n_samples)

# The signal we want to denoise.
f = x ** 2 + 0.3 * np.sin(6 * np.pi * x)

alpha = 0.75

plt.figure()

plt.plot(x, f, lw=3)
plt.plot(x, smooth_signal(f, sigma=0.25), lw=3, label=r"$\sigma=0.25$",
         alpha=alpha)
plt.plot(x, smooth_signal(f, sigma=0.5), lw=3, label=r"$\sigma=0.5$",
         alpha=alpha)
plt.plot(x, smooth_signal(f, sigma=1.0), lw=3, label=r"$\sigma=1.0$",
         alpha=alpha)
plt.xlim((-3, 3))
plt.ylim((-0.5, 10))

plt.legend(loc="best", fontsize=16)

plt.show()
