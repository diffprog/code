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
Decoder-only transformer, minimalist functional implementation in JAX.

Based on:
  https://github.com/karpathy/nanoGPT/blob/master/model.py
  https://github.com/cgarciae/nanoGPT-jax/blob/master/model.py
"""

import math

import jax.numpy as jnp
import jax.scipy as jsp
import jax

# B: Batch size
# L: Maximum sequence length
# D: Embedding size
# H: Number of heads
# C: Dimensionality of matrices W_Q, W_K, W_V
# V: Vocabulary size

def layer_norm(X, weight=None, bias=None, eps=1e-5):
  # X: B x L x C
  # weight: C
  # bias: C

  # Mean and variance over the last axis.
  mean = jnp.mean(X, axis=-1)[:, :, None]
  var = jnp.var(X, axis=-1)[:, :, None]
  new_X = (X - mean) / jnp.sqrt(var + eps)

  if weight is not None:
    new_X *= weight

  if bias is not None:
    new_X += bias

  return new_X  # B x L x C


def causal_self_attention(X, W_Q, W_K, W_V, H):
  # Input
  B, L, C = X.shape

  # Queries, Keys, Values: B x L x C
  Q = X @ W_Q
  K = X @ W_K
  V = X @ W_V
  Q = Q.reshape(B, L, H, C // H).swapaxes(1, 2) # B x H x L x D
  K = K.reshape(B, L, H, C // H).swapaxes(1, 2) # B x H x L x D
  V = V.reshape(B, L, H, C // H).swapaxes(1, 2) # B x H x L x D

  # Attention: B x H x L x L
  A = (Q @ K.swapaxes(-2, -1)) * (1.0 / math.sqrt(K.shape[-1]))

  # Masked attention
  mask = jnp.tril(jnp.ones((L, L)))
  mask = mask.reshape((1, 1, L, L))
  A = jnp.where(mask == 0, float('-inf'), A)
  A = jax.nn.softmax(A, axis=-1)

  # Output
  Y = A @ V # (B x H x L x L) x (B x H x L x D) -> (B x H x L x D)
  Y = Y.swapaxes(1, 2).reshape(B, L, C)
  return Y


def gelu(u):
  return u * jsp.stats.norm.cdf(u)


def mlp(X, W1, W2):
  X = X @ W1
  X = gelu(X)
  X = X @ W2
  return X


def block(X, ln1_params, attn_params, mlp_params, ln2_params):
  X = X + causal_self_attention(layer_norm(X, **ln1_params), **attn_params)
  X = X + mlp(layer_norm(X, **ln2_params), **mlp_params)
  return X


def decoder_transformer(idx, T, P, block_params, ln_params):
  # T: token embedding vectors, V x C
  # P: position embedding vectors, L x C
  B, L = idx.shape

  # Convert sequence of integers to vectors.
  T_emb = T[idx]  # B x L x C
  P_emb = P[jnp.arange(L)]  # L x C
  X = T_emb + P_emb  # B x L x C

  for params in block_params:
    X = block(X, **params)  # B x L x C

  X = layer_norm(X, **ln_params)  # B x L x C

  # Compute output head using weight tying
  # https://paperswithcode.com/method/weight-tying
  logits = X @ T.transpose()  # B x L x V

  return logits
