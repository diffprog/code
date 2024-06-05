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


def and_op(pi1, pi2):
  return pi1 * pi2


def or_op(pi1, pi2):
  return pi1 + pi2 - pi1 * pi2


def plot_logic(op, title):
  x = np.linspace(0, 1, 100)
  y = np.linspace(0, 1, 100)
  X, Y = np.meshgrid(x, y)
  Z = op(X, Y)

  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                  cmap='winter', edgecolor='none')
  ax.set_title(title, fontsize=18)
  fig.tight_layout()


plot_logic(and_op, "And operator")
plot_logic(or_op, "Or operator")

plt.show()
