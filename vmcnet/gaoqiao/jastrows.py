# Copyright 2020 DeepMind Technologies Limited.
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

"""Multiplicative Jastrow factors."""

import enum
from typing import Any, Callable, Iterable, Mapping, Union
from typing_extensions import Protocol
import attr
import jax.numpy as jnp
import chex
import jax
import vmcnet.gaoqiao.network_blocks as network_blocks


ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
  """Available multiplicative Jastrow factors."""

  NONE = enum.auto()
  SIMPLE_EE = enum.auto()
  MLP = enum.auto()


class JastrowInit(Protocol):
  def __call__(self, key, *args) -> ParamTree:
    pass


class JastrowApply(Protocol):
  def __call__(self, Params, *args) -> jnp.ndarray:
    pass

@attr.s(auto_attribs=True)
class JastrowModel:
  init: JastrowInit
  apply: JastrowApply


def _jastrow_ee(
    r_ee: jnp.ndarray,
    params: ParamTree,
    nspins: tuple[int, int],
    jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
  """Jastrow factor for electron-electron cusps."""
  r_ees = [
      jnp.split(r, nspins[0:1], axis=1)
      for r in jnp.split(r_ee, nspins[0:1], axis=0)   #(ne,ne)->[[(n_up,n_up),(n_up,n_down)],[(n_down,n_up),(n_down,n_down)]]
  ]
  r_ees_parallel = jnp.concatenate([
      r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],  #jnp.triu_indices(nspins[0], k=1)生成(n_up,n_up)的上三角矩阵的索引，不包括对角线
      r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
  ]) #得到一个长的一维数组，里面包含了所有(n_up,n_up)和(n_down,n_down)的上三角矩阵的元素

  if r_ees_parallel.shape[0] > 0:
    jastrow_ee_par = jnp.sum(
        jastrow_fun(r_ees_parallel, 0.25, params['ee_par'])
    )
  else:
    jastrow_ee_par = jnp.asarray(0.0)

  if r_ees[0][1].shape[0] > 0:
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees[0][1], 0.5, params['ee_anti']))
  else:
    jastrow_ee_anti = jnp.asarray(0.0)

  return jastrow_ee_anti + jastrow_ee_par


def make_simple_ee_jastrow(
    nspins: tuple[int, int],
    ) -> ...:
  """Creates a Jastrow factor for electron-electron cusps."""

  def simple_ee_cusp_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    """Jastrow function satisfying electron cusp condition."""
    return -(cusp * alpha**2) / (alpha + r)

  def init(
    key: chex.PRNGKey,
    dims_orbital_in:int,
    include_bias:bool=False,
  ):
    del key, include_bias, dims_orbital_in
    params = {}
    params['ee_par'] = jnp.ones(
        shape=1,
    )
    params['ee_anti'] = jnp.ones(
        shape=1,
    )
    return params

  def apply(
      params: ParamTree,
      r_ee: jnp.ndarray,
      he: jnp.ndarray,
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    del he
    return jnp.exp(_jastrow_ee(r_ee, params, nspins, jastrow_fun=simple_ee_cusp_fun)/sum(nspins))

  return JastrowModel(init, apply)


def make_mlp_jastrow(
    nspins,
    hiddenlayers_num:int,
    hiddenlayers_size:int,
    activation_fn:Callable[[jnp.ndarray], jnp.ndarray],
    residual:bool=False,
      ) -> ...:
  """MLP jastrow built from the last layer of eletron feature"""
  def init(
      key: chex.PRNGKey,
      dims_orbital_in:int,
      include_bias:bool=False,
  ) :
    params = []
    key, subkey = jax.random.split(key)
    params.append(
      network_blocks.init_linear_layer(
        subkey,
        in_dim=dims_orbital_in,
        out_dim=hiddenlayers_size,
        include_bias=include_bias,
      ))
    for ii in range(hiddenlayers_num-1):
      key, subkey = jax.random.split(key)
      params.append(
        network_blocks.init_linear_layer(
          subkey,
          in_dim=hiddenlayers_size,
          out_dim=hiddenlayers_size,
          include_bias=include_bias,
        ))
    params.append(
      network_blocks.init_linear_layer(
        subkey,
        in_dim=hiddenlayers_size,
        out_dim=1,
        include_bias=include_bias,
      ))
    return params

  def apply(
      params: ParamTree,
      r_ee: jnp.ndarray,
      he: jnp.ndarray, #last layer of electron hidden states (ne,nfeature)
  )-> jnp.ndarray:
    del r_ee
    jastrow = he
    for ii in range(len(params)-1):
      jastrow_f = activation_fn(network_blocks.linear_layer(jastrow, **params[ii]))
      if residual:
        jastrow = (jastrow + jastrow_f)/jnp.sqrt(2.0) if jastrow_f.shape[1] == jastrow.shape[1] else jastrow_f
      else :
        jastrow = jastrow_f
    jastrow = network_blocks.linear_layer(jastrow, **params[-1])
    jastrow = jnp.sum(jastrow)
    return jastrow

  return JastrowModel(init, apply)

def make_null_jastrow():
  return JastrowModel(None,None)


# def get_jastrow(jastrow: JastrowType) -> ...:
#   jastrow_init, jastrow_apply = None, None
#   if jastrow == JastrowType.SIMPLE_EE:
#     jastrow_init, jastrow_apply = make_simple_ee_jastrow()
#   elif jastrow == JastrowType.MLP:
#     jastrow_init, jastrow_apply = make_mlp_jastrow()
#   elif jastrow != JastrowType.NONE:
#     raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')

#   return jastrow_init, jastrow_apply
