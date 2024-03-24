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
Inference on chains.
"""

import numpy as np
from scipy.special import softmax as softargmax
from scipy.special import logsumexp
from itertools import product


def cartesian_product(k, m):
  for s in product(range(m), repeat=k):
    yield s


def chain_log_sum(pi, s):
  K = pi.shape[0]

  # /!\ The book uses 1-based indexing while Python uses 0-based indexing.
  #
  # p_1(s_1 | s_0) x p_2(s_2 | s_1) x p_3(s_3 | s_2) + ...
  # <->
  # pi[0, 0, s[0]] x pi[1, s[0], s[1]] x pi[2, s[1], s[2]) x ...

  ret = np.log(pi[0, 0, s[0]])

  for k in range(1, K):
    ret += np.log(pi[k, s[k-1], s[k]])

  return ret


def chain_product(pi, s):
  return np.exp(chain_log_sum(pi, s))


def chain_marginal(psi):
  K = psi.shape[0]
  M = psi.shape[1]
  ret = np.zeros((K, M))
  Z = chain_partition(psi)
  for s in cartesian_product(K, M):
    p = chain_product(psi, s) / Z
    for k in range(K):
      ret[k, s[k]] += p
  return ret


def chain_marginal2(psi):
  K = psi.shape[0]
  M = psi.shape[1]
  ret = np.zeros((K, M, M))
  Z = chain_partition(psi)
  for s in cartesian_product(K, M):
    p = chain_product(psi, s) / Z
    for k in range(1, K):
      ret[k, s[k-1], s[k]] += p
  return ret


def chain_partition(psi):
  K = psi.shape[0]
  M = psi.shape[1]
  ret = 0
  for s in cartesian_product(K, M):
    ret += chain_product(psi, s)
  return ret


def chain_mode(psi):
  K = psi.shape[0]
  M = psi.shape[1]

  max_value = -np.inf
  argmax = None

  for s in cartesian_product(K, M):
    p = chain_product(psi, s)
    if p >= max_value:
      max_value = p
      argmax = s

  return max_value, argmax


def forward(psi):
  K = psi.shape[0]
  M = psi.shape[1]

  alpha = np.zeros((K, M))
  alpha[0] = psi[0, 0]

  for k in range(1, K):
    for i in range(M):
      for j in range(M):
        alpha[k, j] += psi[k, i, j] * alpha[k-1, i]

  return alpha


def backward(psi):
  K = psi.shape[0]
  M = psi.shape[1]

  beta = np.zeros((K, M))

  beta[-1] = 1.0

  for k in reversed(range(0, K-1)):
    for i in range(M):
      for j in range(M):
        beta[k, i] += psi[k+1, i, j] * beta[k+1, j]

  return beta


def viterbi(psi):
  K = psi.shape[0]
  M = psi.shape[1]

  delta = np.zeros((K, M))
  q = np.zeros((K, M), dtype=int)
  delta[0] = psi[0, 0]

  for k in range(1, K):
      for j in range(M):
        delta[k, j] = np.max(psi[k, :, j] * delta[k-1])
        q[k, j] = np.argmax(psi[k, :, j] * delta[k-1])

  s_star = np.zeros(K, dtype=int)
  s_star[-1] = delta[-1].argmax()

  for k in reversed(range(0, K-1)):
    s_star[k] += q[k+1, s_star[k+1]]

  return delta[-1].max(), s_star


def argmax(u):
  i = np.argmax(u)
  e = np.zeros(len(u))
  e[i] = 1
  return e


def viterbi_maxop(theta, maxop, argmaxop):
  K = theta.shape[0]
  M = theta.shape[1]

  a = np.ones((K, M)) * (-np.inf)
  a[0] = theta[0, 0]
  q = np.zeros((K, M, M))

  for k in range(1, K):
    for j in range(M):
      a[k, j] = maxop(theta[k, :, j] + a[k-1])
      q[k, j] = argmaxop(theta[k, :, j] + a[k-1])

  mu = np.zeros((K, M, M))
  r = np.zeros((K, M))

  A = maxop(a[-1])
  Q = argmaxop(a[-1])
  r[-1] = Q

  for k in reversed(range(0, K-1)):
    for i in range(M):
      for j in range(M):
        mu[k+1, i, j] = r[k+1, j] * q[k+1, j, i]
        r[k, i] += mu[k+1, i, j]

  return A, mu


def test_forward_backward(K=3, M=4):
  rng = np.random.RandomState(0)
  theta = rng.randn(K, M, M)
  psi = np.exp(theta)

  alpha = forward(psi)
  beta = backward(psi)

  Z_K = np.dot(alpha[K-1], beta[K-1])
  Z_1 = np.dot(alpha[0], beta[0])

  np.testing.assert_almost_equal(Z_1, chain_partition(psi))
  np.testing.assert_almost_equal(Z_1, Z_K)

  np.testing.assert_array_almost_equal(chain_marginal(psi),
                                       alpha * beta / Z_1)


def test_viterbi(K=5, M=4):
  rng = np.random.RandomState(0)
  theta = rng.randn(K, M, M)
  psi = np.exp(theta)

  max_value, argmax = chain_mode(psi)
  max_value2, argmax2 = viterbi(psi)

  np.testing.assert_almost_equal(max_value, max_value2)
  np.testing.assert_array_equal(argmax, argmax2)


def test_viterbi_hard_max(K=4, M=3):
  rng = np.random.RandomState(0)
  theta = rng.randn(K, M, M)
  psi = np.exp(theta)

  max_value, _ = viterbi(psi)
  log_delta, _ = viterbi_maxop(theta, np.max, argmax)
  max_value2 = np.exp(log_delta)

  np.testing.assert_almost_equal(max_value, max_value2)


def test_viterbi_soft_max(K=4, M=3):
  rng = np.random.RandomState(0)
  theta = rng.randn(K, M, M)
  psi = np.exp(theta)

  # Compute marginals using backprop.
  log_Z, mu = viterbi_maxop(theta, logsumexp, softargmax)

  # Compute marginals using forward-backward.
  alpha = forward(psi)
  beta = backward(psi)
  Z = np.sum(alpha[K-1])

  mu2 = np.zeros((K, M, M))
  for k in range(1, K):
    for i in range(M):
      for j in range(M):
        mu2[k, i, j] = alpha[k-1,i] * psi[k, i, j] * beta[k, j] / Z

  # Compute marginals using brute force.
  mu3 = chain_marginal2(psi)

  # Compute marginals by finite difference.
  eps = 1e-6
  mu4 = np.zeros_like(mu3)
  for k in range(1, K):
    for i in range(M):
      for j in range(M):
        theta_pert = theta.copy()
        theta_pert[k, i, j] += eps
        log_Z_pert, _ = viterbi_maxop(theta_pert, logsumexp, softargmax)
        mu4[k, i, j] = (log_Z_pert - log_Z) / eps

  np.testing.assert_almost_equal(Z, np.exp(log_Z))
  np.testing.assert_array_almost_equal(mu2, mu3)
  np.testing.assert_array_almost_equal(mu, mu3)
  np.testing.assert_array_almost_equal(mu3, mu4)


if __name__ == '__main__':
  test_forward_backward()
  test_viterbi()
  test_viterbi_hard_max()
  test_viterbi_soft_max()
