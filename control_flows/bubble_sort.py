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
from scipy.special import expit as logistic


def ifelse(p, v1, v0):
  return p * v1 + (1 - p) * v0


def swap(arr, i, j):
  e_i = np.zeros(len(arr))
  e_i[i] = 1

  e_j = np.zeros(len(arr))
  e_j[j] = 1

  return arr + e_i * (arr[j] - arr[i]) + e_j * (arr[i] - arr[j])


def bubble_sort(arr, gamma=1.0):
  n = len(arr)

  for i in range(n):
    for j in range(0, n - i - 1):
      swapped = swap(arr, j, j+1)
      #p = int(arr[j] > arr[j + 1])
      p = logistic((arr[j] - arr[j + 1]) / gamma)
      arr = ifelse(p, swapped, arr)

  return arr

arr = np.array([3, -1, 1, -2, 5, 4])

print("Original array:")
print(arr)
print()

print("Sorted array array:")
arr= bubble_sort(arr, gamma=0.1)
print(arr)
print()
