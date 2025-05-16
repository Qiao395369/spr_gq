import jax
import jax.numpy as jnp
from typing import Mapping, Optional
import vmcnet.gaoqiao.network_blocks as network_blocks

def self_attn(
    nfeat,
    otdim,
    qkdim: int = 32,
    nhead: int = 2,
    do_gate: bool = False,
    do_lnorm: bool = False,
    lnorm_epsilon: float = 1e-5,
):
  def init(
      key,
      head_scale: float = 1.0
  ):
    params = {}
    key, subkey0, subkey1, subkey2 = jax.random.split(key, 4)
    params['qparam'] = network_blocks.init_linear_layer(subkey0, nfeat, qkdim*nhead, include_bias=False, )
    params['kparam'] = network_blocks.init_linear_layer(subkey1, nfeat, qkdim*nhead, include_bias=False, )
    params['vparam'] = network_blocks.init_linear_layer(subkey2, nfeat, otdim*nhead, include_bias=False, )
    key, subkey0 = jax.random.split(key)
    params['headw'] = network_blocks.init_linear_layer(subkey0, otdim*nhead, otdim, include_bias=False, scale=head_scale)
    if do_gate:
      key, subkey0 = jax.random.split(key)
      params['gate'] = network_blocks.init_linear_layer(subkey0, nfeat, otdim*nhead, )    
    return params

  def apply_layer_norm(
      xx: jnp.ndarray,
      epsilon: float = 1e-5,
  ):
    avg = jnp.average(xx, axis=-1, keepdims=True)
    std = jnp.std(xx, axis=-1, keepdims=True)
    xx = (xx - avg) / (epsilon + std)
    return xx

  def apply(
      params: Mapping[str, jnp.ndarray],
      xx: jnp.ndarray,       # nx x nfeat
      use_vmap: bool = False,
  ):
    qparam = params['qparam']
    kparam = params['kparam']
    vparam = params['vparam']
    headw = params['headw']
    nx = xx.shape[0]
    if do_lnorm:
      xx = apply_layer_norm(xx, epsilon=lnorm_epsilon)
    # nx x qkdim x nh
    qq = network_blocks.vmap_linear_layer(xx, qparam['w'], None).reshape(nx, qkdim, nhead)
    kk = network_blocks.vmap_linear_layer(xx, kparam['w'], None).reshape(nx, qkdim, nhead)
    # nx x otdim x nh
    vv = network_blocks.vmap_linear_layer(xx, vparam['w'], None).reshape(nx, otdim, nhead)
    def apply_map_one_head(qq, kk, vv):
      """
      nx x dk, nx x dk, nx x otdim -> nx x otdim
      """
      # nx x nx
      amap = jnp.matmul(qq, kk.T) / (qkdim**0.5)  #（nx,dk)·(dk,nx)->(nx,nx)
      amap = jax.nn.softmax(amap, axis=1)  #(nx,nx)
      # nx x otdim
      outv = jnp.matmul(amap, vv)  #(nx,nx)·(nx,otdim)->(nx,otdim)
      return outv    
    
    if use_vmap:
        apply_map = jax.vmap(apply_map_one_head, in_axes=(-1,-1,-1), out_axes=(-1))
    else:
        def apply_map(qq, kk, vv):
          """
          nx x dk x h, nx x dk x h, nx x otdim x h -> nx x otdim x h
          """
          amap = jnp.einsum("xdh,ydh->xyh", qq, kk)/ jnp.sqrt(qkdim) # (nx, nx, h)
          amap = jax.nn.softmax(amap, axis=1)
          outv = jnp.einsum("xyh,yoh->xoh", amap, vv) # (nx, otdim, h)
          return outv
 
    # nx x (otdim x nh)
    outv = apply_map(qq, kk, vv).reshape(nx, otdim*nhead)
    if do_gate:
      gate = jax.nn.sigmoid(network_blocks.vmap_linear_layer(xx, params['gate']['w'], params['gate']['b']))
      outv = gate * outv      
    # nx x otdim
    outv = network_blocks.vmap_linear_layer(outv, headw['w'], None)
    return outv

  return init, apply

if __name__=='__main__':
    jax.config.update("jax_enable_x64", True)

    n = 100
    h = 32
    init_fn, apply_fn = self_attn(h, h)
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (n, h))
    
    params = init_fn(key)

    y1 = apply_fn(params, x, True)
    y2 = apply_fn(params, x, False)
    
    assert jnp.allclose(y1, y2)
