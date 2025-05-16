import jax
import jax.numpy as jnp
from typing import Mapping, Optional
import vmcnet.gaoqiao.network_blocks as network_blocks

vmap2_linear_layer = \
  jax.vmap(
    jax.vmap(network_blocks.linear_layer, in_axes=(0, None, None), out_axes=0),
    in_axes=(0, None, None), out_axes=0,
  )

def tri_mul(
    nfeat: int,
    otdim: int,
    nchnl: Optional[int] = None,
    mode: str = "outgoing"
):
  split_abgg = [nchnl, 2*nchnl, 3*nchnl]
  if nchnl is None:
    nchnl = nfeat

  def init(
      key,
  ):
    params = {}
    key, subkey0, subkey1, subkey2 = jax.random.split(key, 4)
    params['x2abgg'] = network_blocks.init_linear_layer(subkey0, nfeat, 4*nchnl, )
    params['ab2oo'] = network_blocks.init_linear_layer(subkey1, nchnl, otdim, )
    params['x2gate'] = network_blocks.init_linear_layer(subkey2, nfeat, otdim, )    
    return params
  
  def apply(
      params,
      xx,
  ):
    """
    xx: npart x npart x nf
    """
    npart = xx.shape[0]
    abgg = vmap2_linear_layer(xx, params['x2abgg']['w'], params['x2abgg']['b'])    #(np,np,nf)->(np,np,4*nchnl)
    aa, bb, ga, gb = jnp.split(abgg, split_abgg, axis=-1)  #四个(np,np,nchnl)
    # np x np x nc
    aa = jax.nn.sigmoid(ga) * aa
    bb = jax.nn.sigmoid(gb) * bb
    # np x np x nc
    if mode == "outgoing":
      ab = jnp.einsum("ikc,jkc->ijc", aa, bb) / float(npart)
    elif mode == "incoming":
      ab = jnp.einsum("kic,kjc->ijc", aa, bb) / float(npart)
    elif mode == "both":
      ab = (jnp.einsum("ikc,jkc->ijc", aa, bb) + 
            jnp.einsum("kic,kjc->ijc", aa, bb) ) \
            / float(npart) * 0.5
    else:
      raise RuntimeError(f"unknow mode {mode}")
    # np x np x no
    oo = vmap2_linear_layer(ab, params['ab2oo']['w'], params['ab2oo']['b'])
    # np x np x no
    gate = jax.nn.sigmoid(vmap2_linear_layer(xx, params['x2gate']['w'], params['x2gate']['b']))
    return gate * oo

  return init, apply
