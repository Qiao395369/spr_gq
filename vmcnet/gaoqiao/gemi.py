# Author:Qiaoqiao
# Date:2024/3/23
# filename:gemi
# Description:
import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing import (
  Sequence, Mapping, Optional, Tuple
)
import vmcnet.gaoqiao.network_blocks as network_blocks


def make_gemi(
    nspins,
    odim: int=32,
    init_style: str = "normal",
    diag_shift: Optional[float] = 0.1,
    weight_dim: int = 0,
):
  #nh1: 16
  #ndet: 1
  #odim: 16
  #init_style: normal
  #diag_shift: 0.1
  #numb_k: 1
  #hiddens: []
  #weight_dim: 1

  def init(
      key: chex.PRNGKey,
  ):
    params = {}
    # weight
    if nspins[1] == 0:
      params['weight'] = None
    else:
      if init_style == "invsqrtio":
        scale = 1./jnp.sqrt(odim+odim)
      elif init_style == "invsqrti":
        scale = 1./jnp.sqrt(odim)
      elif init_style == "invio":
        scale = 1./float(odim+odim)
      elif init_style == "invi":
        scale = 1./float(odim)
      elif init_style == "normal":
        scale = 1.
      else:
        raise RuntimeError(f"unknown init style {init_style}")

      if weight_dim == 2:
        params['weight'] = jax.random.normal(key, shape=(odim,odim)) * scale
      elif weight_dim == 1:   #this
        params['weight'] = jax.random.normal(key, shape=(odim,1)) * scale  #(16,1)
      elif weight_dim == 0:
        params['weight'] = None
      else:
        raise RuntimeError(f'invalide weight_dim {weight_dim}')
      if diag_shift is not None:
        if weight_dim == 2:
          params['weight'] += diag_shift * jnp.eye(odim)
        elif weight_dim == 1:
          params['weight'] += diag_shift

    return params


  def apply_xw(xx, ww):
    """
    xx: cmplx, ne x no
    ww: cmplx, no x ...
    """
    ne, no = xx.shape
    # 2ne x no
    xxc = jnp.concatenate([xx.real, xx.imag], axis=0)
    # 2ne x ...
    ret = network_blocks.vmap_linear_layer(xxc, ww, None)
   # print('ret:',ret)
    #ret=jnp.tanh(ret)
    #print('ret:',ret)
    retr, reti = jnp.split(ret, [ne], axis=0)
    # ne x ...
    return retr + 1.j * reti

  def loggemi(params, x0, x1):
    #x0和x1就是算法第5行的Phi_i:(14,16)的拆分

    #x0:(7,16)or(n1,odim+spdiff)
    #x1:(7,16)or(n2,odim)
    #env:(7,7)or(n1,n1)

    spdiff = nspins[0] - nspins[1]   #0
    # nele0 x nf, nele0 x spdiff
    x00, x01 = jnp.split(x0, [odim], axis=-1)
    assert x00.shape[1] == odim    #16
    assert x01.shape[1] == spdiff  #0
    if weight_dim == 2:
      # nele0 x no
      gij = apply_xw(x00, params['weight'])
      # nele0 x nele1
      gij = jnp.matmul(gij, jnp.conjugate(jnp.transpose(x1, (1,0))))
      # gij = jnp.einsum('iu,uv,jv->ij', x00, params['weight'], jnp.conjugate(x1))
    elif weight_dim == 1:
      ne1, no = x1.shape
      ne0, _  = x0.shape
      # nele0 x nele1 x nf
      xx = jnp.einsum('iu,ju->iju', x00, jnp.conjugate(x1))   #(7,16),(7,16)->(7,7,16)
      #这里得到的就是算法第6行的右边，其中mu就是16维的意思，delta是有几个行列式的意思，就是ndet，在下面vmap中体现。

      # (nele0 x nele1) x nf
      xx = jnp.reshape(xx, [-1, no])  #(49,16)
      #这里是为了运算复数
      #print('xx:',xx[0][0])
      # (nele0 x nele1) x 1
      gij = apply_xw(xx, params['weight'])    #allpy_xw是对复数的线性运算
      #这里就是对算法第6行右边的以w为权重对mu维求和，即(49,16)->(49,1)

      gij = jnp.reshape(gij, [ne0, ne1])
      #再拆分变回复数形式，得到就是算法中的Dij了(这里尚没加上n1,n2之差的那一部分)
      #print('gij:',gij[0][0])
      # gij = jnp.einsum('iu,u,ju->ij', x00, params['weight'], jnp.conjugate(x1))
    elif weight_dim == 0:
      gij = jnp.matmul(x00, jnp.transpose(jnp.conjugate(x1), (1,0)))
    else:
      raise RuntimeError(f'invalide weight_dim {weight_dim}')

    # nele0 x nele0
    gij = jnp.concatenate([gij, x01], axis=-1)
    #补上差的部分，到这里得到的就是算法中的Dij:(n1,n1)!!!!!!!!!!!!!
    #print('gij:',gij) #(8,7,7)
    #print('gij.shape:',gij.shape) #(7,7)
    sign, logdet = jnp.linalg.slogdet(gij)
    #print('logdet:',logdet)
    return sign, logdet

  if nspins[1] == 0:
    loggemi = network_blocks.slogdet
    vmaped_loggemi = jax.vmap(loggemi, in_axes=(0), out_axes=(0))
  else:
    vmaped_loggemi = jax.vmap(loggemi, in_axes=(None,0,0), out_axes=(0,0))
      #这里就是用于计算ndet个Eij*Dij的det和sign

  def apply(
      params: Mapping[str, jnp.ndarray],
      xs: Sequence[jnp.ndarray],
      do_complex: bool = False,
  ):
    #for x in xs:
      #print('x.shape:',x.shape)
    #xs[0]=jnp.tanh(xs[0])
    #xs[1]=jnp.tanh(xs[1])
    if nspins[1] == 0:
      assert len(xs) == 1
      xtmp = xs[0] * env if env is not None else xs[0]
      sign_in, logdet = vmaped_loggemi(xtmp)

    else:
      assert len(xs) == 2
      sign_in,logdet=vmaped_loggemi(
        params,
        xs[0],  #(1,7,16)
        xs[1],  #(1,7,16)
      )
     # print('nonvmap_logdet:',logdet)
      #sign_in, logdet = vmaped_loggemi(
        #params,
       # xs[0],   #(1,7,16)
       # xs[1],   #(1,7,16)
       # )     #(1,7,7)
     # print('logdet:',logdet)
  #得到ndet个Eij*Dij的det和sign

    #下面就是将ndet个det求和，得到最终的log｜Psi｜，和Psi的相位。
    maxlogdet = jnp.max(logdet)
    det = sign_in * jnp.exp(logdet - maxlogdet)
    result = jnp.sum(det)
    if not do_complex:
      sign_out = jnp.sign(result)
    else:
      sign_out = jnp.exp(1j * jnp.angle(result))
    log_out = jnp.log(jnp.abs(result)) + maxlogdet
    return sign_out, log_out

  return init, apply
