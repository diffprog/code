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


def kl(p, q):
  p = np.array(p)
  q = np.array(q)
  ret = 0
  for i in range(len(p)):
    if p[i] == 0 or q[i] == 0:
      ret += q[i]
    else:
      ret += p[i] * np.log(p[i] / q[i]) - p[i] + q[i]
  return ret


def lower_bound(p, q):
  p = np.array(p)
  q = np.array(q)
  return 0.5 * np.sum(np.abs(p - q)) ** 2


pi_vals = np.linspace(0, 1, 100)
q = np.array([0.3, 0.7])

plt.figure()
plt.plot(pi_vals, [kl([pi, 1 - pi], q) for pi in pi_vals], lw=4,
         label=r"$KL(p,q)$")
plt.plot(pi_vals, [lower_bound([pi, 1 - pi], q) for pi in pi_vals], lw=4,
         label=r"$0.5 ||p - q||_1^2$")
plt.xlabel(r"$\pi$", fontsize=17)
plt.legend(loc="best", fontsize=17)
plt.show()


