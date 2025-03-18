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

import torch
from torch.nn import functional as F

from decoder_transformer import block
from decoder_transformer import causal_self_attention
from decoder_transformer import decoder_transformer
from decoder_transformer import layer_norm
from decoder_transformer import mlp

from nanogpt import Block
from nanogpt import CausalSelfAttention
from nanogpt import GPT
from nanogpt import GPTConfig
from nanogpt import MLP


B = 2  # Batch size
L = 5  # Maximum sequence length
D = 12  # Embedding size, must be divisible by H
H = 4  # Number of heads
V = 10  # Vocabulary size

config = GPTConfig(n_embd=D, n_head=H, block_size=L, vocab_size=V, bias=False)


def get_layer_norm_params(ln):
  return dict(weight=ln.weight.detach().numpy())


def test_layer_norm():
  X = torch.randn(B, L, D)
  weight = torch.randn(D)
  bias = torch.randn(D)
  eps = 1e-5
  LN = F.layer_norm(X, weight.shape, weight, bias, eps)
  LN2 = layer_norm(X.numpy(), weight.numpy(), bias.numpy(), eps)
  np.testing.assert_array_almost_equal(LN.numpy(), LN2)


def get_attention_params(attn):
  W = attn.c_attn.weight.detach().numpy()
  return dict(W_Q=W[:D].T,
              W_K=W[D:2*D].T,
              W_V=W[2*D:3*D].T,
              H=attn.n_head)


def test_causal_self_attention():
  attn = CausalSelfAttention(config)
  params = get_attention_params(attn)

  X = torch.randn(B, L, D)
  Y = attn.forward(X)
  Y2 = causal_self_attention(X.numpy(), **params)

  np.testing.assert_array_almost_equal(Y.detach().numpy(), Y2)


def get_mlp_params(model):
  return dict(W1=model.c_fc.weight.detach().numpy().T,
              W2=model.c_proj.weight.detach().numpy().T)


def test_mlp():
  X = torch.randn(B, L, D)

  model = MLP(config)
  params = get_mlp_params(model)

  Y = model.forward(X)
  Y2 = mlp(X.numpy(), **params)

  np.testing.assert_array_almost_equal(Y.detach().numpy(), Y2)


def get_block_params(model):
  return dict(ln1_params=get_layer_norm_params(model.ln_1),
              attn_params=get_attention_params(model.attn),
              mlp_params=get_mlp_params(model.mlp),
              ln2_params=get_layer_norm_params(model.ln_2))


def test_block():
  X = torch.randn(B, L, D)

  model = Block(config)
  params = get_block_params(model)

  Y = model.forward(X)
  Y2 = block(X.numpy(), **params)

  np.testing.assert_array_almost_equal(Y.detach().numpy(), Y2)


def get_gpt_params(model):
  return dict(T=model.transformer.wte.weight.detach().numpy(),
              P=model.transformer.wpe.weight.detach().numpy(),
              block_params=[get_block_params(b) for b in model.transformer.h],
              ln_params=get_layer_norm_params(model.transformer.ln_f))


def test_decoder_transformer():
  model = GPT(config)
  params = get_gpt_params(model)

  idx = torch.randint(V, (B, L))
  logits = model.forward(idx)
  logits2 = decoder_transformer(idx, **params)

  np.testing.assert_array_almost_equal(logits.detach().numpy(), logits2)


if __name__ == '__main__':
  test_layer_norm()
  test_causal_self_attention()
  test_mlp()
  test_block()
  test_decoder_transformer()
  print("Tests passed.")
