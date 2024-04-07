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
import matplotlib.pyplot as plt


# First implementation: closed form.
@np.vectorize
def huber(u, eps=1.0):
  if np.abs(u) <= eps:
    return 0.5 / eps * u ** 2
  else:
    return np.abs(u) - 0.5 * eps


def soft_threshold(u, eps=1.0):
  if np.abs(u) <= eps:
    return 0
  else:
    return u - eps * np.sign(u)


# Second implementation: via Moreau envelope (primal).
@np.vectorize
def huber2(u, eps=1.0):
  x = soft_threshold(u, eps)
  return np.abs(x) + 0.5 / eps * (x - u) ** 2


# Third implementation: via conjugate (dual).
@np.vectorize
def huber3(u, eps=1.0):
  y = max(-1, min(1, u / eps))
  return u * y - 0.5 * eps * y ** 2


u = np.linspace(-3, 3, 500)

fig = plt.figure()
plt.plot(u, huber(u, eps=1.0), label="Huber loss", lw=4, c="C4")
plt.plot(u, np.abs(u), label="Absolute loss", lw=4, c="C2")
plt.legend(loc="best", fontsize=17)
fig.set_tight_layout(True)
plt.show()
