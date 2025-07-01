# Author:Qiaoqiao
# Date:2024/3/23
# filename:network_blocks
# Description:
# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural network building blocks."""

import functools
import itertools
from typing import Mapping, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp


def array_partitions(sizes: Sequence[int]) -> Sequence[int]:
  """Returns the indices for splitting an array into separate partitions.

  Args:
    sizes: size of each of N partitions. The dimension of the array along
    the relevant axis is assumed to be sum(sizes).

  Returns:
    sequence of indices (length len(sizes)-1) at which an array should be split
    to give the desired partitions.
  """
  return list(itertools.accumulate(sizes))[:-1]


def init_linear_layer(key: chex.PRNGKey,
                      in_dim: int,
                      out_dim: int,
                      include_bias: bool = True,
		      scale: float = 1.0,
) -> Mapping[str, jnp.ndarray]:
  """Initialises parameters for a linear layer, x w + b.

  Args:
    key: JAX PRNG state.
    in_dim: input dimension to linear layer.
    out_dim: output dimension (number of hidden units) of linear layer.
    include_bias: if true, include a bias in the linear layer.

  Returns:
    A mapping containing the weight matrix (key 'w') and, if required, bias
    unit (key 'b').
  """
  key1, key2 = jax.random.split(key)
  weight = (
      jax.random.normal(key1, shape=(in_dim, out_dim)) /
      jnp.sqrt(float(in_dim)))
  if include_bias:
    bias = jax.random.normal(key2, shape=(out_dim,))
    return {'w': weight*scale, 'b': bias*scale}
  else:
    return {'w': weight*scale}


def linear_layer(x: jnp.ndarray,
                 w: jnp.ndarray,
                 b: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Evaluates a linear layer, x w + b.

  Args:
    x: inputs.
    w: weights.
    b: optional bias.

  Returns:
    x w + b if b is given, x w otherwise.
  """
  y = jnp.dot(x, w)
  return y + b if b is not None else y

vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)


def slogdet(x):
  """Computes sign and log of determinants of matrices.

  This is a jnp.linalg.slogdet with a special (fast) path for small matrices.

  Args:
    x: square matrix.

  Returns:
    sign, (natural) logarithm of the determinant of x.
  """
  if x.shape[-1] == 1:
    sign = jnp.sign(x[..., 0, 0])
    logdet = jnp.log(jnp.abs(x[..., 0, 0]))
  else:
    sign, logdet = jnp.linalg.slogdet(x)

  return sign, logdet


def logdet_matmul(xs: Sequence[jnp.ndarray],
                  w: Optional[jnp.ndarray] = None,
                  do_complex: bool = False,
                  RFM_layer: int=0,
                  RFM_w: Optional[jnp.ndarray] = None,
                  RFM_numbers: Optional[Tuple[int, ...]] = None
                  ) -> jnp.ndarray:
  """Combines determinants and takes dot product with weights in log-domain.

  We use the log-sum-exp trick to reduce numerical instabilities.

  Args:
    xs: FermiNet orbitals in each determinant. Either of length 1 with shape
      (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
      (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
      determinants are factorised into block-diagonals for each spin channel).
    w: weight of each determinant. If none, a uniform weight is assumed.

  Returns:
    sum_i w_i D_i in the log domain, where w_i is the weight of D_i, the i-th
    determinant (or product of the i-th determinant in each spin channel, if
    full_det is not used).
  """
  # 1x1 determinants appear to be numerically sensitive and can become 0
  # (especially when multiple determinants are used with the spin-factored
  # wavefunction). Avoid this by not going into the log domain for 1x1 matrices.
  # Pass initial value to functools so det1d = 1 if all matrices are larger than
  # 1x1.
  det1d = functools.reduce(lambda a, b: a * b,
                           [x.reshape(-1) for x in xs if x.shape[-1] == 1], 1)
  # Pass initial value to functools so sign_in = 1, logdet = 0 if all matrices
  # are 1x1.
  sign_in, logdet = functools.reduce(
      lambda a, b: (a[0] * b[0], a[1] + b[1]),
      [slogdet(x) for x in xs if x.shape[-1] > 1], (1, 0))

  # log-sum-exp trick
  maxlogdet = jnp.max(logdet)  #logdet.shape:(ndet,)
  det = sign_in * det1d * jnp.exp(logdet - maxlogdet)  #sign_in.shape:(ndet,)  det.shape:(ndet,)

  if w is None:
    if RFM_layer != 0 and RFM_w is not None:
      
      det_RFM= jnp.repeat(det, RFM_layer)
      random_RFM=jnp.sin(det_RFM*RFM_numbers)
      result = linear_layer(random_RFM, **RFM_w)[0]
    else:
      result = jnp.sum(det)
  else:
    if isinstance(w, list):
      for i in range(len(w)):
        if i != len(w)-1:
          det_next = jnp.tanh(linear_layer(det, **w[i]))
        else:
          det_next = linear_layer(det, **w[i])
        det = 1./jnp.sqrt(2.) * (det + det_next)
      result = jnp.sum(det)
    else:
      result = jnp.matmul(det, w)

  if not do_complex:
    sign_out = jnp.sign(result)
  else:
    sign_out = jnp.exp(1j * jnp.angle(result))
  log_out = jnp.log(jnp.abs(result)) + maxlogdet
  return sign_out, log_out

#对于(nedt，n，n)的xs，sign_in和logdet分别为(ndet,),(nedt,)的数组，用log-sum-exp trick先把最大的提出来之后再加上
#对于[(ndet,nalpha,nalpha),(ndet,nbeta,nbeta)]的xs，同样得到的是(ndet,),(nedt,)的sign_in和logdet
#[(ndet,nalpha,nalpha),(ndet,nbeta,nbeta)]得到的[slogdet(x) for x in xs if x.shape[-1] > 1]就是一个由两个元组构成的列表
#而(nedt，n，n)得到的是只有一个元组的列表，不管哪一种，元组都是由两个(ndet,)的数组构成的，对应sign_in和logdet.
#之后要么是求和，即对nedt个行列式求和；
#或者在w只是一个(nedt,)的数组时，计算行列式数组与w的点乘得到一个数；
#或者在w是一个由很多(ndet,)构成的列表的时候，对行列式数组进行dense——layer处理。