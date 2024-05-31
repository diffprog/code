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


def shannon_entropy(p):
  p = np.array(p)
  mask = p > 0
  plogp = np.zeros_like(p)
  plogp[mask] = p[mask] * np.log(p[mask])
  return -np.sum(plogp)


def gini_entropy(p):
  return 0.5 * (1 - np.dot(p, p))


def tsallis_entropy(p, alpha=1.5):
  p = np.array(p)
  scale = 1./ (alpha * (alpha - 1))
  return scale * (1 - np.sum(p ** alpha))


unit_segment = np.linspace(0, 1.0, 200)

fig = plt.figure()
plt.plot(unit_segment, [shannon_entropy([v, 1-v]) for v in unit_segment],
         label=r"Tsallis $\alpha \to 1$ (Shannon)", lw=4, c="C2")
plt.plot(unit_segment, [tsallis_entropy([v, 1-v], alpha=1.5)
                        for v in unit_segment],
         label=r"Tsallis $\alpha=1.5$", lw=4, c="C3")
plt.plot(unit_segment, [gini_entropy([v, 1-v]) for v in unit_segment],
         label=r"Tsallis $\alpha=2$ (Gini)", lw=4, c="C4")
plt.legend(loc="best", fontsize=17)
plt.xlabel(r"$\pi$", fontsize=15)
fig.set_tight_layout(True)
plt.show()
