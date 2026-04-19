# Copyright 2023 DeepMind Technologies Limited.
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

"""Attention-based networks for FermiNet."""

from typing import Mapping, Optional, Sequence, Tuple, Union

import attr
import chex
from vmcnet.gaoqiao.fermi_ferminet import fermi_envelopes
import vmcnet.gaoqiao.envelopes as envelopes
from vmcnet.gaoqiao.fermi_ferminet import jastrows
from vmcnet.gaoqiao.fermi_ferminet import fermi_network_blocks as network_blocks
from vmcnet.gaoqiao.fermi_ferminet import fermi_networks as networks
from vmcnet.gaoqiao.dp import reform_ee_ea_ae_aa
import jax
import jax.numpy as jnp
import numpy as np


@attr.s(auto_attribs=True, kw_only=True)
class PsiformerOptions(networks.BaseNetworkOptions):
  """Options controlling the Psiformer part of the network architecture.

  Attributes:
    num_layers: Number of self-attention layers.
    num_heads: Number of multihead self-attention heads.
    heads_dim: Embedding dimension for each self-attention head.
    mlp_hidden_dims: Tuple of sizes of hidden dimension of the MLP. Note that
      this does not include the final projection to the embedding dimension.
    use_layer_norm: If true, include a layer norm on both attention and MLP.
  """

  num_layers: int = 2
  num_heads: int = 4
  heads_dim: int = 64
  mlp_hidden_dims: Tuple[int, ...] = (256,)
  use_layer_norm: bool = False
  psiformer_multi: bool = False


def make_layer_norm() ->...:
  """Implementation of LayerNorm."""

  def init(param_shape: int) -> Mapping[str, jnp.ndarray]:
    params = {}
    params['scale'] = jnp.ones(param_shape)
    params['offset'] = jnp.zeros(param_shape)
    return params

  def apply(params: networks.ParamTree,
            inputs: jnp.ndarray,
            axis: int = -1) -> jnp.ndarray:
    mean = jnp.mean(inputs, axis=axis, keepdims=True)
    variance = jnp.var(inputs, axis=axis, keepdims=True)
    eps = 1e-5
    inv = params['scale'] * jax.lax.rsqrt(variance + eps)
    return inv * (inputs - mean) + params['offset']

  return init, apply


def make_multi_head_attention(num_heads: int, heads_dim: int) ->...:
  """FermiNet-style version of MultiHeadAttention."""

  # Linear layer plus reshape final dimensions to num_heads, heads_dim.
  def linear_projection(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    y = jnp.dot(x, weights)
    return y.reshape(*x.shape[:-1], num_heads, heads_dim)

  def init(key: chex.PRNGKey,
           q_d: int,
           kv_d: int,
           output_channels: Optional[int] = None) -> Mapping[str, jnp.ndarray]:

    # Dimension of attention projection.
    qkv_hiddens = num_heads * heads_dim
    if not output_channels:
      output_channels = qkv_hiddens

    key, *subkeys = jax.random.split(key, num=4)
    params = {}
    params['q_w'] = network_blocks.init_linear_layer(
        subkeys[0], in_dim=q_d, out_dim=qkv_hiddens, include_bias=False)['w']
    params['k_w'] = network_blocks.init_linear_layer(
        subkeys[1], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False)['w']
    params['v_w'] = network_blocks.init_linear_layer(
        subkeys[2], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False)['w']

    key, subkey = jax.random.split(key)
    params['attn_output'] = network_blocks.init_linear_layer(
        subkey, in_dim=qkv_hiddens, out_dim=output_channels,
        include_bias=False)['w']

    return params

  def apply(params: networks.ParamTree, query: jnp.ndarray, key: jnp.ndarray,
            value: jnp.ndarray) -> jnp.ndarray:
    """Computes MultiHeadAttention with keys, queries and values.

    Args:
      params: Parameters for attention embeddings.
      query: Shape [..., q_index_dim, q_d]
      key: Shape [..., kv_index_dim, kv_d]
      value: Shape [..., kv_index_dim, kv_d]

    Returns:
      A projection of attention-weighted value projections.
      Shape [..., q_index_dim, output_channels]
    """

    # Projections for q, k, v.
    # Output shape: [..., index_dim, num_heads, heads_dim].
    q = linear_projection(query, params['q_w'])
    k = linear_projection(key, params['k_w'])
    v = linear_projection(value, params['v_w'])

    attn_logits = jnp.einsum('...thd,...Thd->...htT', q, k)
    scale = 1. / np.sqrt(heads_dim)
    attn_logits *= scale

    attn_weights = jax.nn.softmax(attn_logits)

    attn = jnp.einsum('...htT,...Thd->...thd', attn_weights, v)

    # Concatenate attention matrix of all heads into a single vector.
    # Shape [..., q_index_dim, num_heads * heads_dim]
    attn = jnp.reshape(attn, (*query.shape[:-1], -1))

    # Apply a final projection to get the final embeddings.
    # Output shape: [..., q_index_dim, output_channels]
    return network_blocks.linear_layer(attn, params['attn_output'])

  return init, apply


def make_mlp() ->...:
  """Construct MLP, with final linear projection to embedding size."""

  def init(key: chex.PRNGKey, mlp_hidden_dims: Tuple[int, ...],
           embed_dim: int) -> Sequence[networks.Param]:
    params = []
    dims_one_in = [embed_dim, *mlp_hidden_dims]
    dims_one_out = [*mlp_hidden_dims, embed_dim]
    for i in range(len(dims_one_in)):
      key, subkey = jax.random.split(key)
      params.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_one_in[i],
              out_dim=dims_one_out[i],
              include_bias=True))
    return params

  def apply(params: Sequence[networks.Param],
            inputs: jnp.ndarray) -> jnp.ndarray:
    x = inputs
    for i in range(len(params)):
      x = jnp.tanh(network_blocks.linear_layer(x, **params[i]))
    return x

  return init, apply


def make_self_attention_block(num_layers: int,
                              num_heads: int,
                              heads_dim: int,
                              mlp_hidden_dims: Tuple[int, ...],
                              use_layer_norm: bool = False) ->...:
  """Create a QKV self-attention block."""
  attention_init, attention_apply = make_multi_head_attention(
      num_heads, heads_dim)
  if use_layer_norm:
    layer_norm_init, layer_norm_apply = make_layer_norm()
  mlp_init, mlp_apply = make_mlp()

  def init(key: chex.PRNGKey, qkv_d: int) -> networks.ParamTree:
    attn_dim = qkv_d
    params = {}
    attn_params = []
    ln_params = []
    mlp_params = []

    for _ in range(num_layers):
      key, attn_key, mlp_key = jax.random.split(key, 3)
      attn_params.append(
          attention_init(
              attn_key, q_d=qkv_d, kv_d=qkv_d, output_channels=attn_dim))
      if use_layer_norm:
        ln_params.append([layer_norm_init(attn_dim), layer_norm_init(attn_dim)])
      mlp_params.append(mlp_init(mlp_key, mlp_hidden_dims, attn_dim))

    params['attention'] = attn_params
    params['ln'] = ln_params
    params['mlp'] = mlp_params

    return params

  def apply(params: networks.ParamTree, qkv: jnp.ndarray) -> jnp.ndarray:
    x = qkv
    for layer in range(num_layers):
      attn_output = attention_apply(params['attention'][layer], x, x, x)

      # Residual + optional LayerNorm.
      x = x + attn_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][0], x)

      # MLP
      assert isinstance(params['mlp'][layer], (tuple, list))
      mlp_output = mlp_apply(params['mlp'][layer], x)

      # Residual + optional LayerNorm.
      x = x + mlp_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][1], x)

    return x

  return init, apply


def make_psiformer_layers(
    nspins: Tuple[int, ...],
    natoms: int,
    options: PsiformerOptions,
) -> Tuple[networks.InitLayersFn, networks.ApplyLayersFn]:
  """Creates the permutation-equivariant layers for the Psiformer.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    natoms: number of atoms.
    options: network options.

  Returns:
    Tuple of init, apply functions.
  """
  del nspins  # Unused.

  # Attention network.
  attn_dim = options.num_heads * options.heads_dim
  self_attn_init, self_attn_apply = make_self_attention_block(
      num_layers=options.num_layers,
      num_heads=options.num_heads,
      heads_dim=options.heads_dim,
      mlp_hidden_dims=options.mlp_hidden_dims,
      use_layer_norm=options.use_layer_norm,
  )

  def init(key: chex.PRNGKey) -> Tuple[int, networks.ParamTree]:
    """Returns tuple of output dimension from the final layer and parameters."""
    params = {}
    key, subkey = jax.random.split(key)
    feature_dims, params['input'] = options.feature_layer.init()
    one_electron_feature_dim, _ = feature_dims
    # Concatenate spin of each electron with other one-electron features.
    feature_dim = one_electron_feature_dim + 1

    # Map to Attention dim.
    key, subkey = jax.random.split(key)
    params['embed'] = network_blocks.init_linear_layer(
        subkey, 
        in_dim=feature_dim, 
        out_dim=attn_dim, 
        include_bias=False
        )['w']

    # Attention block params.
    key, subkey = jax.random.split(key)
    params.update(self_attn_init(key, attn_dim))

    return attn_dim, params

  def apply(
      params,
      pos,
      atoms,
      spins: jnp.ndarray,
  ) -> jnp.ndarray:
    """Applies the Psiformer interaction layers to a walker configuration.

    Args:
      params: parameters for the interaction and permuation-equivariant layers.
      ae: electron-nuclear vectors.
      r_ae: electron-nuclear distances.
      ee: electron-electron vectors.
      r_ee: electron-electron distances.
      spins: spin of each electron.
      charges: nuclear charges.

    Returns:
      Array of shape (nelectron, output_dim), where the output dimension,
      output_dim, is given by init, and is suitable for projection into orbital
      space.
    """
    # Only one-electron features are used by the Psiformer.
    h_one, h_two, ae, r_ee = options.feature_layer.apply(pos,atoms,options.ndim)

    # For the Psiformer, the spin feature is required for correct permutation equivariance.
    h_one = jnp.concatenate((h_one, spins[..., None]), axis=-1)

    features = h_one  # Just 1-electron stream for now.

    # Embed into attention dimension.
    x = jnp.dot(features, params['embed'])

    h_to_orbitals = self_attn_apply(params, x)

    if options.psiformer_multi == True:
      h_to_orbitals = h_to_orbitals[:-natoms]
      hz = h_to_orbitals[-natoms:]
    else:
      hz = None

    return h_to_orbitals, hz, ae, r_ee

  return init, apply


def make_fermi_net(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    psiformer_multi: bool = False,
    # Psiformer-specific kwargs below.
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
) -> networks.Network:
  """Psiformer with stacked Self Attention layers.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    heads_dim: Embedding dimension per-head for self-attention.
    mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    raise 

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = make_ferminet_features(
        natoms, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      psiformer_multi=psiformer_multi,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = make_orbitals(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      atoms: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """
    orbitals = orbitals_apply(params, pos, atoms, spins)
    if options.states:
      batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
      orbitals = [
          jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
          for orbital in orbitals
      ]
      result = batch_logdet_matmul(orbitals)
    else:
      result = network_blocks.logdet_matmul(orbitals)
    if 'state_scale' in params:
      # only used at inference time for excited states
      result = result[0], result[1] + params['state_scale']
    return result[:-1]
  
  def network_each_det(
      params,
      pos: jnp.ndarray,
      atoms: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    orbitals = orbitals_apply(params, pos, atoms, spins)
    result = network_blocks.logdet_matmul(orbitals)
    return jax.nn.log_softmax(result[2])
  
  return network_init, network_apply, options, network_each_det, orbitals_apply







def make_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: PsiformerOptions,
    equivariant_layers: Tuple[networks.InitLayersFn, networks.ApplyLayersFn],
) -> ...:

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers
  jastrow = options.jastrow

  def init(key: chex.PRNGKey) -> networks.ParamTree:

    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1)
    for nspin in active_spin_channels:
      if options.full_det:
        # Dense determinant. Need N orbitals per electron per determinant.
        norbitals = sum(nspins) * options.determinants * num_states
      else:
        # Spin-factored block-diagonal determinant. Need nspin orbitals per
        # electron per determinant.
        norbitals = nspin * options.determinants * num_states
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals)
    nspin_orbitals=[int(i) for i in nspin_orbitals]

    # create envelope params
    natom = charges.shape[0]

    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      # Applied to output from final layer of 1e stream.
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # Applied to orbitals.
      if options.complex_output:
        # print("nspin_orbitals:",nspin_orbitals)
        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
      else:
        output_dims = nspin_orbitals
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT_Z:
      output_dims = nspin_orbitals
    else:
      raise ValueError('Unknown envelope type')
    
    if options.envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT_Z:
      params['envelope'] = options.envelope.init(natom=natom, output_dims=output_dims, ndim=options.ndim)
    else:
      key, subkey = jax.random.split(key)
      params['envelope'] = options.envelope.init(subkey, hz_size=dims_orbital_in, output_dims=output_dims)

    if jastrow.init is not None:
      key, subkey = jax.random.split(key)
      params['jastrow'] = jastrow.init(key = subkey, dims_orbital_in = dims_orbital_in)

    orbitals = []
    for nspin_orbital in nspin_orbitals:
      key, subkey = jax.random.split(key)
      orbitals.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_orbital_in,
              out_dim=nspin_orbital,
              include_bias=options.bias_orbitals,
          )
      )
    params['orbital'] = orbitals

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      atoms: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """
    h_to_orbitals_, hz, ae, r_ee = equivariant_layers_apply(params['layers'], pos, atoms, spins=spins)
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)

    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      envelope_factor = options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
      )
      h_to_orbitals = envelope_factor * h_to_orbitals_

    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split( h_to_orbitals_, network_blocks.array_partitions(nspins), axis=0)

    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = network_blocks.array_partitions(active_spin_channels
                                                                   )
    # Create orbitals.
    orbitals = [network_blocks.linear_layer(h, **p) for h, p in zip(h_to_orbitals, params['orbital'])]

    if options.complex_output:
      # create imaginary orbitals
      orbitals = [orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals]

    # Apply envelopes if required.
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
      r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
      r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
      for i in range(len(active_spin_channels)):
        orbitals[i] = orbitals[i] * options.envelope.apply(
            ae=ae_channels[i],
            r_ae=r_ae_channels[i],
            r_ee=r_ee_channels[i],
            **params['envelope'][i],
            )
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT_Z:
      ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
      for i in range(len(active_spin_channels)):
        orbitals[i] = orbitals[i] * options.envelope.apply(
            hz=hz,
            ae=ae_channels[i],
            **params['envelope'][i],     
            ) 

    # Reshape into matrices.
    shapes = [
        (spin, -1, sum(nspins) if options.full_det else spin)
        for spin in active_spin_channels
    ]
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    if options.full_det:
      orbitals = [jnp.concatenate(orbitals, axis=1)]

    # Optionally apply Jastrow factor for electron cusp conditions.
    # Added pre-determinant for compatibility with pretraining.
    if jastrow.apply is not None:
      jastrow_ = jastrow.apply(params['jastrow'], r_ee, h_to_orbitals_) 
      orbitals = [orbital * jastrow_ for orbital in orbitals]

    return orbitals

  return init, apply


def make_ferminet_features_multi(
    natoms: int,
    ndim: int = 3,
    rescale_inputs: bool = False,
) -> networks.FeatureLayer:
  """Returns the init and apply functions for the standard features."""
  assert rescale_inputs==True
  def init() -> Tuple[Tuple[int, int], networks.Param]:
    return (ndim + 1, ndim + 1), {}

  def apply(pos: jnp.ndarray,
            atoms: jnp.ndarray,
            ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:

    assert atoms.shape[1] == ndim
    ea = jnp.reshape(pos, (-1, 1, ndim)) - atoms[None, ...]   #(ne,na,3)
    ee = jnp.reshape(pos, (-1, 1, ndim)) - jnp.reshape(pos, (1, -1, ndim))  #()
    ne = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(ne)[..., None], axis=-1) * (1.0 - jnp.eye(ne)))[...,None]

    pp_pos = jnp.concatenate([pos, atoms],axis=0)
    pp = jnp.reshape(pp_pos, [-1, 1, ndim]) - jnp.reshape(pp_pos, [1, -1, ndim])
    np = pp.shape[0]
    r_pp = (jnp.linalg.norm(pp + jnp.eye(np)[..., None], axis=-1) * (1.0 - jnp.eye(np)))[..., None]

    if rescale_inputs:
      log_r_pp = jnp.log(1 + r_pp)
      factor=jnp.where(r_pp!=0, log_r_pp / r_pp, 0.0)
      pp_features = jnp.concatenate((log_r_pp, pp * factor), axis=2)
    else:
      pp_features = jnp.concatenate((r_pp, pp), axis=2)

    h_one = jnp.mean(pp_features,axis=1)
    h_two = pp_features
    return h_one,h_two,ea,r_ee

  return networks.FeatureLayer(init=init, apply=apply)


def make_ferminet_features(
    natoms: int,
    ndim: int = 3,
    rescale_inputs: bool = False,
) -> networks.FeatureLayer:
  """Returns the init and apply functions for the standard features."""


  def init() -> Tuple[Tuple[int, int], networks.Param]:
    return (natoms * (ndim + 1), ndim + 1), {}

  def apply(pos: jnp.ndarray,
            atoms: jnp.ndarray,
            ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:

    ea = jnp.reshape(pos, (-1, 1, ndim)) - atoms[None, ...]   #(ne,na,3)
    ee = jnp.reshape(pos, (-1, 1, ndim)) - jnp.reshape(pos, (1, -1, ndim))  #()
    r_ea = jnp.linalg.norm(ea, axis=2, keepdims=True)
    ne = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(ne)[..., None], axis=-1) * (1.0 - jnp.eye(ne)))[...,None]

    if rescale_inputs:
      log_r_ea = jnp.log(1 + r_ea)  # grows as log(r) rather than r
      factor=jnp.where(r_ea!=0, log_r_ea / r_ea, 0.0)
      ea_features = jnp.concatenate((log_r_ea, ea * factor), axis=2)

      log_r_ee = jnp.log(1 + r_ee)
      factor=jnp.where(r_ee!=0, log_r_ee / r_ee, 0.0)
      ee_features = jnp.concatenate((log_r_ee, ee * factor), axis=2)
    else:
      ea_features = jnp.concatenate((r_ea, ea), axis=2)
      ee_features = jnp.concatenate((r_ee, ee), axis=2)

    ea_features = jnp.reshape(ea_features, (ne, -1))
    return ea_features, ee_features, ea, r_ee

  return networks.FeatureLayer(init=init, apply=apply)