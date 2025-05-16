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

"""Implementation of Fermionic Neural Network in JAX."""
import enum
import functools
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import attr
import chex
import vmcnet.gaoqiao.envelopes as envelopes
import vmcnet.gaoqiao.network_blocks as network_blocks
# import sto
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
import vmcnet.gaoqiao.dp as dp
import vmcnet.gaoqiao.sto as sto
import vmcnet.gaoqiao.gemi as gemi
import vmcnet.gaoqiao.attn as attn
import vmcnet.gaoqiao.tri as tri

FermiLayers = Tuple[Tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]
# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param = Mapping[str, jnp.ndarray]

## Interfaces (public) ##


class InitFermiNet(Protocol):

  def __call__(self, key: chex.PRNGKey) -> ParamTree:
    """Returns initialized parameters for the network.

    Args:
      key: RNG state
    """


class FermiNetLike(Protocol):

  def __call__(self, params: ParamTree,
               electrons: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
    """


class LogFermiNetLike(Protocol):

  def __call__(self, params: ParamTree, electrons: jnp.ndarray) -> jnp.ndarray:
    """Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
    """

## Interfaces (network components) ##


class FeatureInit(Protocol):

  def __call__(self) -> Tuple[Tuple[int, int], Param]:
    """Creates the learnable parameters for the feature input layer.

    Returns:
      Tuple of ((x, y), params), where x and y are the number of one-electron
      features per electron and number of two-electron features per pair of
      electrons respectively, and params is a (potentially empty) mapping of
      learnable parameters associated with the feature construction layer.
    """


class FeatureApply(Protocol):

  def __call__(self, ae: jnp.ndarray, r_ae: jnp.ndarray, ee: jnp.ndarray,
               r_ee: jnp.ndarray,
               **params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Creates the features to pass into the network.

    Args:
      ae: electron-atom vectors. Shape: (nelectron, natom, 3).
      r_ae: electron-atom distances. Shape: (nelectron, natom, 1).
      ee: electron-electron vectors. Shape: (nelectron, nelectron, 3).
      r_ee: electron-electron distances. Shape: (nelectron, nelectron).
      **params: learnable parameters, as initialised in the corresponding
        FeatureInit function.
    """


@attr.s(auto_attribs=True)
class FeatureLayer:
  init: FeatureInit
  apply: FeatureApply


class ModelInit(Protocol):

  def __call__(self, key) -> Param:
    """Creates the learnable parameters for the model.

    Returns:
      key: the random key
    """


class ModelApply(Protocol):

  def __call__(self, 
               Params,
               ae: jnp.ndarray, 
               ee: jnp.ndarray) -> jnp.ndarray:
    """Creates the model to pass into the network.

    Args:
      params: learnable model parameters
      ae: electron-atom vectors. Shape: (nelectron, natom, x).
      ee: electron-electron vectors. Shape: (nelectron, nelectron, x).
    """


@attr.s(auto_attribs=True)
class FerminetModel:
  init: ModelInit
  apply: ModelApply


class FeatureLayerType(enum.Enum):
  STANDARD = enum.auto()


class MakeFeatureLayer(Protocol):

  def __call__(self,
               charges: jnp.ndarray,
               nspins: Sequence[int],
               ndim: int,
               **kwargs: Any) -> FeatureLayer:
    """Builds the FeatureLayer object.

    Args:
      charges: (natom) array of atom nuclear charges.
      nspins: tuple of the number of spin-up and spin-down electrons.
      ndim: dimension of the system.
      **kwargs: additional kwargs to use for creating the specific FeatureLayer.
    """


## Network settings ##


@attr.s(auto_attribs=True, kw_only=True)
class FermiNetOptions:
  """Options controlling the FermiNet architecture.

  Attributes:
    ndim: dimension of system. Change only with caution.
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
    determinants: Number of determinants to use.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    envelope: Envelope object to create and apply the multiplicative envelope.
    feature_layer: Feature object to create and apply the input features for the
      one- and two-electron layers.
  """
  ndim: int = 3
  hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32), (256, 32))
  use_last_layer: bool = False
  determinants: int = 16
  full_det: bool = True
  bias_orbitals: bool = False
  envelope: envelopes.Envelope = attr.ib(
      default=attr.Factory(
          envelopes.make_isotropic_envelope,
          takes_self=False))
  feature_layer: FeatureLayer = None
  # attr.ib(default=attr.Factory(lambda self: make_ferminet_features(ndim=self.ndim), takes_self=True))
  ferminet_model: FerminetModel = None 
  # attr.ib(default=attr.Factory(lambda self: make_fermi_net_model, takes_self=True))
#   lattice : jnp.array = None
  det_nlayer : int = None
  do_complex : bool = False
  envelope_pw: Any = None
  orb_env_pw: Any = None
  do_twist: bool = False
  do_aa: bool = False
  mes: dp.ManyElectronSystem = None
  det_mode: str = 'det'
  gemi_params: dict = None
  gemi_ia: Any = None
  equal_footing: bool = False


## Network initialisation ##

def init_layers(
    key: chex.PRNGKey, dims_one_in: Sequence[int], dims_one_out: Sequence[int],
    dims_two_in: Sequence[int],
    dims_two_out: Sequence[int]) -> Tuple[Sequence[Param], Sequence[Param]]:
  """Initialises parameters for the FermiNet layers.

  The final two-electron layer is not strictly necessary (i.e.
  FermiNetOptions.use_last_layer is False), in which case the two-electron
  stream contains one fewer layers than the one-electron stream.

  Args:
    key: JAX RNG state.
    dims_one_in: dimension of inputs to each one-electron layer.
    dims_one_out: dimension of outputs (number of hidden units) in each
      one-electron layer.
    dims_two_in: dimension of inputs to each two-electron layer.
    dims_two_out: dimension of outputs (number of hidden units) in each
      two-electron layer.

  Returns:
    Pair of sequences (length: number of layers) of parameters for one- and
    two-electon streams.

  Raises:
    ValueError: if dims_one_in and dims_one_out are different lengths, or
    similarly for dims_two_in and dims_two_out, or if the number of one-electron
    layers is not equal to or one more than the number of two electron layers.
  """
  if len(dims_one_in) != len(dims_one_out):
    raise ValueError(
        'Length of one-electron stream inputs and outputs not identical.')
  if len(dims_two_in) != len(dims_two_out):
    raise ValueError(
        'Length of two-electron stream inputs and outputs not identical.')
  if len(dims_two_in) not in (len(dims_one_out), len(dims_one_out) - 1):
    raise ValueError('Number of layers in two electron stream must match or be '
                     'one fewer than the number of layers in the one-electron '
                     'stream')
  single = []
  double = []
  ndouble_layers = len(dims_two_in)
  for i in range(len(dims_one_in)):
    key, subkey = jax.random.split(key)
    single.append(
        network_blocks.init_linear_layer(
            subkey,
            in_dim=dims_one_in[i],
            out_dim=dims_one_out[i],
            include_bias=True))

    if i < ndouble_layers:
      key, subkey = jax.random.split(key)
      double.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_two_in[i],
              out_dim=dims_two_out[i],
              include_bias=True))

  return single, double


def init_orbital_shaping(
    key: chex.PRNGKey,
    input_dim: int,
    nspin_orbitals: Sequence[int],
    bias_orbitals: bool,
) -> Sequence[Param]:
  """Initialises orbital shaping layer.

  Args:
    key: JAX RNG state.
    input_dim: dimension of input activations to the orbital shaping layer.
    nspin_orbitals: total number of orbitals in each spin-channel.
    bias_orbitals: whether to include a bias in the layer.

  Returns:
    Parameters of length len(nspin_orbitals) for the orbital shaping for each
    spin channel.
  """
  orbitals = []
  for nspin_orbital in nspin_orbitals:
    key, subkey = jax.random.split(key)
    orbitals.append(
        network_blocks.init_linear_layer(
            subkey,
            in_dim=input_dim,
            out_dim=nspin_orbital,
            include_bias=bias_orbitals))
  return orbitals


def init_to_hf_solution(
    hf_solution,
    single_layers: Sequence[Param],
    orbital_layer: Sequence[Param],
    determinants: int,
    active_spin_channels: Sequence[int],
    eps: float = 0.01) -> Tuple[Sequence[Param], Sequence[Param]]:
  """Sets initial parameters to match Hartree-Fock.

  NOTE: this does not handle the envelope parameters, which are done in the
  appropriate envelope initialisation functions. Not all envelopes support HF
  initialisation.

  Args:
    hf_solution: Hartree-Fock state to match.
    single_layers: parameters (weights and biases) for the one-electron stream,
      with length: number of layers in the one-electron stream.
    orbital_layer: parameters for the orbital-shaping layer, length: number of
      spin-channels in the system.
    determinants: Number of determinants used in the final wavefunction.
    active_spin_channels: Number of particles in each spin channel containing at
      least one particle.
    eps: scaling factor for all weights and biases such that they are
      initialised close to zero unless otherwise required to match Hartree-Fock.

  Returns:
    Tuple of parameters for the one-electron stream and the orbital shaping
    layer respectively.
  """
  # Scale all params in one-electron stream to be near zero.
  single_layers = jax.tree_map(lambda param: param * eps, single_layers)
  # Initialize first layer of Fermi Net to match s- or p-type orbitals.
  # The sto and sto-poly envelopes can exactly capture the s-type orbital,
  # so the effect of the neural network part is constant, while the p-type
  # orbital also has a term multiplied by x, y or z.
  j = 0
  for ia, atom in enumerate(hf_solution.molecule):
    coeffs = sto.STO_6G_COEFFS[atom.symbol]
    for orb in coeffs.keys():
      if orb[1] == 's':
        single_layers[0]['b'] = single_layers[0]['b'].at[j].set(1.0)
        j += 1
      elif orb[1] == 'p':
        w = single_layers[0]['w']
        w = w.at[ia * 4 + 1:(ia + 1) * 4, j:j + 3].set(jnp.eye(3))
        single_layers[0]['w'] = w
        j += 3
      else:
        raise NotImplementedError('HF Initialization not implemented for '
                                  f'{orb[1]} orbitals')
  # Scale all params in orbital shaping to be near zero.
  orbital_layer = jax.tree_map(lambda param: param * eps, orbital_layer)
  for i, spin in enumerate(active_spin_channels):
    # Initialize last layer to match Hartree-Fock weights on basis set.
    norb = hf_solution.mean_field.mo_coeff[i].shape[0]
    mat = hf_solution.mean_field.mo_coeff[i][:, :spin]
    w = orbital_layer[i]['w']
    for j in range(determinants):
      w = w.at[:norb, j * spin:(j + 1) * spin].set(mat)
    orbital_layer[i]['w'] = w
  return single_layers, orbital_layer


def init_fermi_net_params(
    key: chex.PRNGKey,
    natom: int,
    ndim: int,
    nspins: Tuple[int, ...],
    options: FermiNetOptions,
    hf_solution = None,
    eps: float = 0.01,
) -> ParamTree:
  """Initializes parameters for the Fermionic Neural Network.

  Args:
    key: JAX RNG state.
    atoms: (natom, ndim) array of atom positions.
    nspins: A tuple with either the number of spin-up and spin-down electrons,
      or the total number of electrons. If the latter, the spins are instead
      given as an input to the network.
    options: network options.
    hf_solution: If present, initialise the parameters to match the Hartree-Fock
      solution. Otherwise a random initialisation is use.
    eps: If hf_solution is present, scale all weights and biases except the
      first layer by this factor such that they are initialised close to zero.

  Returns:
    PyTree of network parameters. Spin-dependent parameters are only created for
    spin channels containing at least one particle.
  """
  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    if options.bias_orbitals:
      raise ValueError('Cannot bias orbitals w/STO envelope.')
  if hf_solution is not None:
    if options.use_last_layer:
      raise ValueError('Cannot use last layer w/HF init')
    if options.envelope.apply_type not in ('sto', 'sto-poly'):
      raise ValueError('When using HF init, envelope_type must be `sto` or `sto-poly`.')

  active_spin_channels = [spin for spin in nspins if spin > 0]
  nchannels = len(active_spin_channels)
  if nchannels == 0:
    raise ValueError('No electrons present!')

  params = {}
  key, subkey = jax.random.split(key, num=2)
  _params, dims_orbital_in = options.ferminet_model.init(subkey)
  for kk, vv in _params.items():
    params[kk] = vv

  # How many spin-orbitals do we need to create per spin channel?
  nspin_orbitals = []
  if options.det_mode == "det":
    for nspin in active_spin_channels:
      if options.full_det:
        # Dense determinant. Need N orbitals per electron per determinant.
        norbitals = sum(nspins) * options.determinants  #最终的nspin_orbitals=[(n_up+n_dn)*ndet,(n_up+n_dn)*ndet]
      else:
        # Spin-factored block-diagonal determinant. Need nspin orbitals per electron per determinant.
        norbitals = nspin * options.determinants      #最终的nspin_orbitals=[n_up*ndet,n_dn*ndet]
      nspin_orbitals.append(norbitals)
  elif options.det_mode == "gemi":
    # assumes that nspins[0] >= nspins[1]
    diff_sp = nspins[0]-nspins[1]  #n_up-n_dn
    if nspins[1] == 0:
      # case nspins = (spin, 0)
      nspin_orbitals.append(diff_sp * options.determinants)      #n_up*ndet
    else:
      # case nspins = (spin0, spin1)
      nspin_orbitals.append((options.gemi_params['odim']+diff_sp) * options.determinants)  #(odim+diff)*ndet
      nspin_orbitals.append((options.gemi_params['odim']) * options.determinants)  # odim*ndet
      #最终的nspin_orbitals=[(odim+diff)*ndet,odim*ndet]
  else:
    raise RuntimeError(f"unknown det_mode {options.det_mode}")

  if options.do_complex:
    # twice the output channels if do_complex
    nspin_orbitals_alloc = [ii*2 for ii in nspin_orbitals]
  else:
    nspin_orbitals_alloc = nspin_orbitals

  # create envelope params
  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    # Applied to output from final layer of 1e stream.
    output_dims = dims_orbital_in
  elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    # Applied to orbitals.
    output_dims = nspin_orbitals
  elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT_Z:
    # Applied to orbitals.
    output_dims = nspin_orbitals
  elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT_TWIST:
    # Applied to orbitals.
    output_dims = nspin_orbitals
  elif options.envelope.apply_type == envelopes.EnvelopeType.POST_DETERMINANT:
    # Applied to all determinants.
    output_dims = 1
  else:
    raise ValueError('Unknown envelope type')
  if options.envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT_Z:
    params['envelope'] = options.envelope.init(
      natom=natom, output_dims=output_dims, hf=hf_solution, ndim=ndim)
  else:
    params['envelope'] = options.envelope.init(
      key, hz_size=options.hidden_dims[-1][0], output_dims=output_dims)

  # orbital shaping
  key, subkey = jax.random.split(key, num=2)
  params['orbital'] = init_orbital_shaping(
      key=subkey,
      input_dim=dims_orbital_in,
      nspin_orbitals=nspin_orbitals_alloc,
      bias_orbitals=options.bias_orbitals)

  if hf_solution is not None:
    params['single'], params['orbital'] = init_to_hf_solution(
        hf_solution=hf_solution,
        single_layers=params['single'],
        orbital_layer=params['orbital'],
        determinants=options.determinants,
        active_spin_channels=active_spin_channels,
        eps=eps)

  if options.det_nlayer is not None and options.det_nlayer > 0:
    params['det'] = []
    for ii in range(options.det_nlayer):
      key, subkey = jax.random.split(key)
      params['det'].append(
        network_blocks.init_linear_layer(
          subkey,
          in_dim=options.determinants,
          out_dim=options.determinants,
          include_bias=False,
        ))
  else:
    params['det'] = None

  if options.det_mode == "gemi":
    key, subkey = jax.random.split(key)
    params['gemi'] = options.gemi_ia[0](subkey)

  if options.envelope_pw is not None:
    key, subkey = jax.random.split(key)
    params['envelope_pw'] = options.envelope_pw.init(
      key, dims_orbital_in, options.hidden_dims[-1][0])

  if options.orb_env_pw is not None:
    key, subkey = jax.random.split(key)
    # backed by nspin_orbitals,
    # because output_dims is twice the nspin_orbitals if complex
    params['orb_env_pw'] = options.orb_env_pw.init(
      key, dims_orbital_in, nspin_orbitals)

  return params

## Network layers ##
def construct_input_features_ef(
    pos_: jnp.ndarray,
    ndim: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Constructs inputs to Fermi Net from raw electron and atomic positions.

  Args:
    pos: electron positions. Shape (nelectrons*ndim,).
    atoms: atom positions. Shape (natoms, ndim).
    ndim: dimension of system. Change only with caution.

  Returns:
    pp, r_pp, where:
      pp: particle-particle vector. Shape (nelectron, natom, ndim).
      r_pp: electron-electron distance. Shape (nelectron, nelectron, 1).
    The diagonal terms in r_ee are masked out such that the gradients of these
    terms are also zero.
  """
  pos = pos_

  ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
  ne = ee.shape[0]
  r_ee = (jnp.linalg.norm(ee + jnp.eye(ne)[..., None], axis=-1) * (1.0 - jnp.eye(ne)))

  return ee, r_ee[..., None]






def construct_symmetric_features(h_one: jnp.ndarray, h_two: jnp.ndarray,
                                 nspins: Tuple[int, int]) -> jnp.ndarray:
  """Combines intermediate features from rank-one and -two streams.

  Args:
    h_one: set of one-electron features. Shape: (nelectrons, n1), where n1 is
      the output size of the previous layer.
    h_two: set of two-electron features. Shape: (nelectrons, nelectrons, n2),
      where n2 is the output size of the previous layer.
    nspins: Number of spin-up and spin-down electrons.

  Returns:
    array containing the permutation-equivariant features: the input set of
    one-electron features, the mean of the one-electron features over each
    (occupied) spin channel, and the mean of the two-electron features over each
    (occupied) spin channel. Output shape (nelectrons, 3*n1 + 2*n2) if there are
    both spin-up and spin-down electrons and (nelectrons, 2*n1 + n2) otherwise.
  """
  # Split features into spin up and spin down electrons
  spin_partitions = network_blocks.array_partitions(nspins)
  h_ones = jnp.split(h_one, spin_partitions, axis=0)
  h_twos = jnp.split(h_two, spin_partitions, axis=0)

  # Construct inputs to next layer
  # h.size == 0 corresponds to unoccupied spin channels.
  g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
  g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]

  g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]

  return jnp.concatenate([h_one] + g_one + g_two, axis=1)


def make_fermi_net_model_ef(
    natom,
    ndim,
    nspins,
    feature_layer,
    hidden_dims,   #i.e. ((64,16),(64,16),(64,16),(64,16))
    use_last_layer = False,
    dim_extra_params = 0,
    do_aa : bool=False,
    mes = None,
    # extra parameters
    layer_update_scheme: Optional[dict] = None,
    attn_params: Optional[dict] = None,
    trimul_params: Optional[dict] = None,
    reduced_h1_size: Optional[int] = None,
    h1_attn_params: Optional[dict] = None,
):
  del do_aa,
  # do_aa should always be true
  assert (dim_extra_params==0),"dim_extra_params should be 0 in gq "
  do_aa = True
  if mes is None :
    raise RuntimeError('make_fermi_net_model_ef only support equal-footing models')
  
  if layer_update_scheme is not None:
    update_alpha = layer_update_scheme.get("update_alpha", None)
    do_rdt = layer_update_scheme.get("do_resd_dt", False)
    if do_rdt:
      resd_dt_shift = layer_update_scheme.get("resd_dt_shift", 1.0)
      resd_dt_scale = layer_update_scheme.get("resd_dt_scale", 0.1)
    else:
      resd_dt_shift, resd_dt_scale = None, None
  else:
    update_alpha, do_rdt, resd_dt_shift, resd_dt_scale = None, False, None, None

  # atten on two particle channel
  do_attn = attn_params is not None
  if do_attn:
    attn_nfeat = hidden_dims[-1][1]
    attn_init, attn_apply = attn.self_attn(attn_nfeat, attn_nfeat,**attn_params,)
    # input is npart x npart x nf, vmap along axis==1
    vmap_attn_apply = jax.vmap(attn_apply, in_axes=(None,1), out_axes=1,)

  do_trimul = trimul_params is not None
  if do_trimul:
    trimul_nfeat = hidden_dims[-1][1]
    trimul_nchnl = trimul_params.get("nchnl", 8)
    trimul_mode = trimul_params.get("mode", "both")
    trimul_init, trimul_apply = tri.tri_mul(trimul_nfeat, trimul_nfeat, nchnl=trimul_nchnl, mode=trimul_mode)

  dh_scale = sum([do_attn, do_trimul])
  if dh_scale != 0:
    dh_scale = 1./float(dh_scale)

  do_h1_attn = h1_attn_params is not None
  if do_h1_attn:
    h1_attn_params = dict(h1_attn_params)
    h1_attn_nfeat = hidden_dims[-1][0]
    h1_attn_init, h1_attn_apply = attn.self_attn(h1_attn_nfeat, h1_attn_nfeat,**h1_attn_params,)

  def _init_resd_dt(
      key,
      layer_size: list,
  ):
    ret = []
    for ii in layer_size:
      key, subkey = jax.random.split(key)
      ret.append(resd_dt_shift + resd_dt_scale * jax.random.normal(subkey, shape=(ii,)))
    return ret

  def init(
      key,
  ):    
    dim_posi_code = mes.get_dim_one_hot()    #3
    dim_pair_code = 2*mes.get_dim_one_hot()   #6

    # extra params, dim of rs is 1
    # dim_1_append = dim_extra_params + 1 + dim_posi_code  
    # dim_2_append = dim_extra_params + 1 + dim_pair_code
    # ｜
    # ｜
    # V
    #no rs in gq_job
    dim_1_append = dim_extra_params + dim_posi_code    #3
    dim_2_append = dim_extra_params + dim_pair_code    #6

    # number of spin channel
    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)  #2
    # init params
    params = {}
    (num_one_features, num_two_features), params['input'] = (feature_layer.init())
    distinguish_ele = True
    if not distinguish_ele:  #distinguish_ele=True , which means distinguish different spins.
      if reduced_h1_size is None:
        nfeatures = lambda out1, out2: 2 * out1 + 2 * out2
      else:
        nfeatures = lambda out1, out2: out1 + min(reduced_h1_size,out1) + 2 * out2
    else:
      if reduced_h1_size is None:
        nfeatures = lambda out1, out2: (nchannels+1) * out1 + (nchannels+1) * out2  # 3*xxx + 3*xxx
      else:
        nfeatures = lambda out1, out2: out1 + nchannels * min(reduced_h1_size,out1) + (nchannels+1) * out2  
        #ou1+2*min(out1,reduced_h1_size) + 3*out2
    dims_1_in = [nfeatures(hidden_dims[0][0], hidden_dims[0][1])]     # nfeatures(64,16)
    dims_1_in += [nfeatures(hdim[0], hdim[1]) for hdim in hidden_dims[:-1]]  
    #dims_1_in=[nfeatures(64,16), nfeatures(64,16), nfeatures(64,16), nfeatures(64,16)]
    dims_1_out = [hdim[0] for hdim in hidden_dims]  # [64,64,64,64]
    if use_last_layer:  #use_last_layer=False
      dims_2_in = ([hidden_dims[0][1]] + [hdim[1] for hdim in hidden_dims[:-1]])  # [16,16,16,16]
      dims_2_out = [hdim[1] for hdim in hidden_dims]  # [16,16,16,16]
    else:
      dims_2_in = ([hidden_dims[0][1]] + [hdim[1] for hdim in hidden_dims[:-2]])  # [16,16,16]
      dims_2_out = [hdim[1] for hdim in hidden_dims[:-1]]  # [16,16,16]
    # Layer initialisation
    key, subkey = jax.random.split(key)
    params['one'], params['two'] = init_layers(
        key=subkey,
        dims_one_in=dims_1_in,  # [nfeatures(64,16), nfeatures(64,16), nfeatures(64,16), nfeatures(64,16)]
        dims_one_out=dims_1_out,  # [64,64,64,64]
        dims_two_in=dims_2_in,  # [16,16,16]
        dims_two_out=dims_2_out)  # [16,16,16]
    key, k1, k2 = jax.random.split(key, 3)    
    params['one_dt'] = _init_resd_dt(k1, dims_1_out) if do_rdt else None
    params['two_dt'] = _init_resd_dt(k2, dims_2_out) if do_rdt else None
    if not use_last_layer:
      dims_orbital_in = hidden_dims[-1][0]  #64
    else:
      dims_orbital_in = nfeatures(hidden_dims[-1][0], hidden_dims[-1][1]) 

    # reshape h1, h2 to hidden_dims[0][0] and hidden_dims[0][1] at the very first layer so the update is via residule
    key, subkey = jax.random.split(key) 
    dim_one_reshp_in = num_one_features + dim_1_append  #4+3
    dim_one_reshp_out = hidden_dims[0][0]
    params['one_reshp'] = network_blocks.init_linear_layer(subkey, dim_one_reshp_in, dim_one_reshp_out, )  #7->64
    del dim_one_reshp_in, dim_one_reshp_out
    key, subkey = jax.random.split(key)
    dim_two_reshp_in = num_two_features + dim_2_append  #4+6
    dim_two_reshp_out = hidden_dims[0][1]
    params['two_reshp'] = network_blocks.init_linear_layer(subkey, dim_two_reshp_in, dim_two_reshp_out, )  #10->16
    del dim_two_reshp_in, dim_two_reshp_out

    # projection parameters
    dim_proj_1_in = dims_1_out[:len(dims_2_out)]  #[64,64,64]
    dim_proj_1_out = dims_2_out  # [16,16,16]
    params['proj'] = []
    for ii in range(len(params['two'])):
      if dim_proj_1_in[ii] != dim_proj_1_out[ii]:
        key, subkey = jax.random.split(key)
        params['proj'].append(network_blocks.init_linear_layer(subkey, dim_proj_1_in[ii], dim_proj_1_out[ii], ))
        #[64 , 64 , 64]
        # |    |    |
        # |    |    |
        # V    V    V
        #[16 , 16 , 16]
      else:
        params['proj'].append(None)  # do not project if the input and output dims are the same
    dim_proj_0_in = hidden_dims[0][0]  #64
    dim_proj_0_out = hidden_dims[0][1]  #16
    if dim_proj_0_in != dim_proj_0_out:
      key, subkey = jax.random.split(key)
      params['proj_0'] = network_blocks.init_linear_layer(subkey, dim_proj_0_in, dim_proj_0_out, )  #64->16
    else:
      params['proj_0'] = None

    if do_attn:
      params['attn'] = []
      for i in range(len(params['two'])):
        key, subkey = jax.random.split(key)
        params['attn'].append(attn_init(subkey))
      key, subkey = jax.random.split(key)
      params['attn_dt'] = _init_resd_dt(subkey, dims_2_out) if do_rdt else None

    if do_trimul:
      params['trimul'] = []
      for i in range(len(params['two'])):
        key, subkey = jax.random.split(key)
        params['trimul'].append(trimul_init(subkey))
      key, subkey = jax.random.split(key)
      params['trimul_dt'] = _init_resd_dt(subkey, dims_2_out) if do_rdt else None

    if do_h1_attn:
      params['h1_attn'] = []
      for i in range(len(params['one'])):
        key, subkey = jax.random.split(key)
        head_scale = 1.0
        params['h1_attn'].append(h1_attn_init(subkey, head_scale=head_scale))
      key, subkey = jax.random.split(key)
      params['h1_attn_dt'] = _init_resd_dt(subkey, dims_1_out) if do_rdt else None

    return params, dims_orbital_in

  def construct_symmetric_features_conv(
          h1 : jnp.ndarray,
          h2 : jnp.ndarray,
          proj: Optional[Mapping[str,jnp.ndarray]] = None,
  ) -> jnp.ndarray:
    """
    hi  : np x nfi
    hij : np x np x nfij
    """
    if proj is not None:
      ph1 = network_blocks.linear_layer(h1, **proj)  #(np,nf_two)
    else:
      ph1 = h1
    h2xh1 = h2 * ph1[:,None,:]   # (np,np,nf_two)* (np,1,nf_two)->(np,np,nf_two)  p_h1 *(pointwise product) h2
    hijxhi, hizxhi, hzixhz, hyzxhy = mes.split_ee_ea_ae_aa(h2xh1)
    hi, hz = mes.split_ea(h1)
    distinguish_ele = True
    if distinguish_ele:
      spin_partitions = network_blocks.array_partitions(nspins)
      # [nele x nfij, nele x nfij]
      hij_i = [jnp.mean(h, axis=0) for h in jnp.split(hijxhi, spin_partitions, axis=0) if h.size > 0]
      # nele x nfiz
      hzi_i = [jnp.mean(hzixhz, axis=0)]
      # [nz x nfiz, nz x nfiz]
      hiz_z = [jnp.mean(h, axis=0) for h in jnp.split(hizxhi, spin_partitions, axis=0) if h.size > 0]
      # [nz x nfyz]
      hyz_z = [jnp.mean(hyzxhy, axis=0)] if do_aa else []
      # 1 x nfiz
      gz = jnp.mean(hz, axis=0, keepdims=1)
      # nz x nfiz
      gz = jnp.tile(gz, [hz.shape[0], 1])
      # [1 x nfij, 1 x nfij]
      gi = [jnp.mean(h, axis=0, keepdims=1) for h in jnp.split(hi, spin_partitions, axis=0) if h.size > 0]
      # [nele x nfij, nele x nfij]
      gi = [jnp.tile(g, [hi.shape[0], 1]) for g in gi]
      # nz x nfiz, nz x nfiz
      gz = [gz, gz] if len(gi) == 2 else [gz]
      if reduced_h1_size is not None:
        gi = [jnp.split(ii, [min(reduced_h1_size,ii.shape[-1])], axis=-1)[0] for ii in gi]  
        gz = [jnp.split(ii, [min(reduced_h1_size,ii.shape[-1])], axis=-1)[0] for ii in gz]
        #截断，只要前min(reduced_h1_size,ii.shape[-1])的特征
    return \
      jnp.concatenate([
        jnp.concatenate([hi] + gi + hij_i + hzi_i, axis=-1),
        jnp.concatenate([hz] + gz + hiz_z + hyz_z, axis=-1),
      ], axis=0)

  def _hi_next(
      hi_in,
      params,
  ):
    return jnp.tanh(network_blocks.linear_layer(hi_in, **params))

  def _hij_next(
      hij_in,
      params,
  ):
    return jnp.tanh(network_blocks.vmap_linear_layer(hij_in,params['w'],params['b'],))

  def apply(
      params,
      e2_features,
  ):

    c1 = mes.get_part_one_hot()  #(nele+nz,3)
    c2 = mes.get_pair_one_hot()  #(nele+nz,nele+nz,6)

    a1 = c1
    a2 = c2
    
    def residual(x, y):      
      if update_alpha is None:
        if not do_rdt :
          return (x + y) / jnp.sqrt(2.0)
        else:
          return x + y
      else:
        return update_alpha * x + jnp.sqrt(1. - update_alpha**2) * y  #平方和为1

    npart = e2_features.shape[0]
    h2 = e2_features
    hee, _, haa = mes.split_ee_ea_aa(h2)
    ha = jnp.mean(haa, axis=0)  #\Sigma_x hxy : (na,na,nf_two)->(na,nf_two)
    he = jnp.mean(hee, axis=0)  #\Sigma_i hij :(ne,ne,nf_two)->(ne,nf_two)
    h1 = jnp.concatenate([he, ha], axis=0)    #(ne+na,nf_two)
    for i in range(len(params['two'])):
      if i == 0:
        h1 = jnp.concatenate([h1, a1], axis=-1)  # (ne+na,nf_two+3)
        h2 = jnp.concatenate([h2, a2], axis=-1)  # (ne+na,ne+na,nf_two+6)
        h1 = network_blocks.linear_layer(h1, **params['one_reshp'])  # (ne+na,nf_two+3)->(ne+na,64)
        h2 = network_blocks.linear_layer(h2, **params['two_reshp'])  # (ne+na,ne+na,nf_two+6)->(ne+na,ne+na,16)
        h1_in = construct_symmetric_features_conv(h1, h2, params['proj_0'])
      else:
        h1_in = construct_symmetric_features_conv(h1, h2, params['proj'][i-1])
      # channel one
      h1_next = _hi_next(h1_in, params['one'][i])
      h1_next = params['one_dt'][i] * h1_next if do_rdt else h1_next
      # channel two
      h2_next = _hij_next(h2, params['two'][i])
      h2_next = params['two_dt'][i] * h2_next if do_rdt else h2_next
      # update
      # np x nh1
      h1 = residual(h1, h1_next)
      # np x np x nh2
      h2 = residual(h2, h2_next)
      # h1_attn
      if do_h1_attn:
        h1_delta = h1_attn_apply(params['h1_attn'][i], h1)
        h1_delta = params['h1_attn_dt'][i] * h1_delta if do_rdt else h1_delta
        h1 = residual(h1, h1_delta)
      # h2_attn and h2_trimul
      h2_delta = jnp.zeros_like(h2)
      if do_attn:
        h2_delta_attn = dh_scale*vmap_attn_apply(params['attn'][i], h2)
        h2_delta += params['attn_dt'][i] * h2_delta_attn if do_rdt else h2_delta_attn
      if do_trimul:
        h2_delta_trim = dh_scale*trimul_apply(params['trimul'][i], h2)
        h2_delta += params['trimul_dt'][i] * h2_delta_trim if do_rdt else h2_delta_trim
      h2 = residual(h2, h2_delta)
    if len(params['two']) != len(params['one']):
      h1_in = construct_symmetric_features_conv(h1, h2, params['proj'][-1])
      h1_next = _hi_next(h1_in, params['one'][-1])
      h1_next = params['one_dt'][-1] * h1_next if do_rdt else h1_next
      h1 = residual(h1, h1_next)
      if do_h1_attn:
        h1_delta = h1_attn_apply(params['h1_attn'][i], h1)
        h1_delta = params['h1_attn_dt'][i] * h1_delta if do_rdt else h1_delta
        h1 = residual(h1, h1_delta)
      h_to_orbitals = h1
    else:
      _, h_to_orbitals = construct_symmetric_features_conv(h1, h2, params['proj'][-1])
    return h_to_orbitals, h1

  return FerminetModel(init, apply)


def fermi_net_orbitals(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):
  """Forward evaluation of the Fermionic Neural Network up to the orbitals.

  Args:
    params: A dictionary of parameters, containing fields:
      `atoms`: atomic positions, used to construct input features.
      `single`: a list of dictionaries with params 'w' and 'b', weights for the
        one-electron stream of the network.
      `double`: a list of dictionaries with params 'w' and 'b', weights for the
        two-electron stream of the network.
      `orbital`: a list of two weight matrices, for spin up and spin down (no
        bias is necessary as it only adds a constant to each row, which does not
        change the determinant).
      `dets`: weight on the linear combination of determinants
      `envelope`: a dictionary with fields `sigma` and `pi`, weights for the
        multiplicative envelope.
    pos: The electron positions, a 3N dimensional vector.
    atoms: Array with positions of atoms.
    nspins: Tuple with number of spin up and spin down electrons.
    options: Network configuration.

  Returns:
    One matrix (two matrices if options.full_det is False) that exchange columns
    under the exchange of inputs of shape (ndet, nalpha+nbeta, nalpha+nbeta) (or
    (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)) and a tuple of (ae, r_ae,
    r_ee), the atom-electron vectors, distances and electron-electron distances.
  """
  assert (options.equal_footing),"equal_footing should be True in gq"
  h_to_orbitals, hz, pairs = fermi_net_orbitals_part1_ef(params, pos, atoms, nspins, options)
  ret = fermi_net_orbitals_part2(params, pos, h_to_orbitals, hz, pairs, nspins, options)
  return ret

def fermi_net_orbitals_part1_ef(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):
  """
  """
  pos = jnp.concatenate([pos.reshape(-1), atoms.reshape(-1)])  #(ne+na,3)
  pp, r_pp = construct_input_features_ef(pos)  # (ne+na,ne+na,3),(ne+na,ne+na,1)
  pp_features = options.feature_layer.apply(ee=pp, r_ee=r_pp, **params['input'])  # (ne+na,ne+na,nf_two)
  hpp = pp_features
  h_to_orbitals, hp = options.ferminet_model.apply(params, hpp,)
  [h_to_orbitals, _] = options.mes.split_ea(h_to_orbitals)  #(ne,64)
  [ee, ae, aa] = options.mes.split_ee_ea_aa(pp)
  [r_ee, r_ae, r_aa] = options.mes.split_ee_ea_aa(r_pp)
  [hi, hz] = options.mes.split_ea(hp)  #(ne,64),(na,64)
  return h_to_orbitals, hz, (ee, ae, aa, r_ee, r_ae, r_aa)


def fermi_net_orbitals_part2(
    params,
    pos,
    h_to_orbitals,
    hz,
    pair_dist,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):
  model_h_to_orbitals = h_to_orbitals
  (ee, ae, aa, r_ee, r_ae, r_aa) = pair_dist

  # Note split creates arrays of size 0 for spin channels without any electrons.
  h_to_orbitals = jnp.split(h_to_orbitals, network_blocks.array_partitions(nspins), axis=0) #(nele,64)->[(n_up,64),(n_dn,64)]
  # Drop unoccupied spin channels
  h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]  # [(n_up,64),(n_dn,64)]
  active_spin_channels = [spin for spin in nspins if spin > 0]    #[n_up,n_dn]
  active_spin_partitions = network_blocks.array_partitions(active_spin_channels)  #[n_up]
  # Create orbitals.
  orbitals = [network_blocks.linear_layer(h, **p)for h, p in zip(h_to_orbitals, params['orbital'])]  
  #i.e. fulldet_type_orbitals and not do_complex:  [(n_up,64),(n_dn,64)]->[(n_up,nele*ndet),(n_dn,nele*ndet)]
  #i.e. fulldet_type_orbitals and do_complex:  [(n_up,64),(n_dn,64)]->[(n_up,nele*ndet*2),(n_dn,nele*ndet*2)]
  #i.e. not fulldet and not do_complex:  [(n_up,64),(n_dn,64)]->[(n_up,n_up*ndet),(n_dn,n_dn*ndet)]

  # if do_complex, make the orbitals complex numbers
  if options.do_complex:
    for ii in range(len(active_spin_channels)):
      nsplit = orbitals[ii].shape[-1] // 2
      split_orb = jnp.split(orbitals[ii], [nsplit], axis=-1)
      orbitals[ii] = split_orb[0] + 1j * split_orb[1]
    #[(n_up,nele*ndet*2),(n_dn,nele*ndet*2)]->[(n_up,nele*ndet),(n_dn,nele*ndet)]

  # Apply envelopes if required.
  assert (options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT_Z),"envelope_apply_type should be PRE_DETERMINANT_Z in gq"
  ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
  for i in range(len(active_spin_channels)):
      orbitals[i] = orbitals[i] * options.envelope.apply(
      hz=hz,
      ae=ae_channels[i],
      **params['envelope'][i],
      )    


  # Reshape into matrices.
  if options.det_mode == "det":
    shapes = [(spin, -1, sum(nspins) if options.full_det else spin) for spin in active_spin_channels]
    #full_det: [(n_up,-1,nele),(n_dn,-1,nele)]  else: [(n_up,-1,n_up),(n_dn,-1,n_dn)]
  else:
    shapes = [(spin, options.determinants, -1) for spin in active_spin_channels] # [(n_up,ndet,-1),(n_dn,ndet,-1)]
  orbitals = [jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)]  
  #full_det: [(n_up,nele*ndet),(n_dn,nele*ndet)]->[(n_up,ndet,nele),(n_dn,ndet,nele)] 
  #not full_det: [(n_up,n_up*ndet),(n_dn,n_dn*ndet)]->[(n_up,ndet,n_up),(n_dn,ndet,n_dn)]
  orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
  #full_det: [(n_up,ndet,nele),(n_dn,ndet,nele)]->[(ndet,n_up,nele),(ndet,n_dn,nele)]
  #not full_det: [(n_up,ndet,n_up),(n_dn,ndet,n_dn)]->[(ndet,n_up,n_up),(ndet,n_dn,n_dn)]
  if options.full_det:
    orbitals = [jnp.concatenate(orbitals, axis=1)] #[(ndet,nele,nele)]
  #本质上，ndet=1时，对于输入的orbitals_in:(nele,64)->[(n_up,64),(n_dn,64)]->[(n_up,nele),(n_dn,nele)]->[(nele,nele)] 
  #等效于：第一维是nele个轨道，第二维是nele个电子填充。
  return orbitals, (ae, r_ae, r_ee, model_h_to_orbitals, hz)

## FermiNet ##


def fermi_net(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):
  """Forward evaluation of the Fermionic Neural Network for a single datum.

  Args:
    params: A dictionary of parameters, containing fields:
      `atoms`: atomic positions, used to construct input features.
      `single`: a list of dictionaries with params 'w' and 'b', weights for the
        one-electron stream of the network.
      `double`: a list of dictionaries with params 'w' and 'b', weights for the
        two-electron stream of the network.
      `orbital`: a list of two weight matrices, for spin up and spin down (no
        bias is necessary as it only adds a constant to each row, which does not
        change the determinant).
      `dets`: weight on the linear combination of determinants
      `envelope`: a dictionary with fields `sigma` and `pi`, weights for the
        multiplicative envelope.
    pos: The electron positions, a 3N dimensional vector.
    atoms: Array with positions of atoms.
    nspins: Tuple with number of spin up and spin down electrons.
    options: network options.

  Returns:
    Output of antisymmetric neural network in log space, i.e. a tuple of sign of
    and log absolute of the network evaluated at x.
  """

  orbitals, (ae, r_ae, r_ee, he, hz) = fermi_net_orbitals(
      params,
      pos,
      atoms=atoms,
      nspins=nspins,
      options=options,
  )
  assert (options.envelope_pw is None),"envelope_pw should be None in gq"

  if params['det'] is not None:
    w = params['det']
  else:
    w = None

  output = network_blocks.logdet_matmul(orbitals, w=w, do_complex=options.do_complex)

  return output


def make_fermi_net(
    natom: int, 
    ndim:int, 
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    *,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[FeatureLayer] = None,
    ferminet_model: Optional[FerminetModel] = None,
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    hf_solution = None,
    full_det: bool = True,
    lattice: Optional[jnp.ndarray] = None,
    mes: dp.ManyElectronSystem = None,
    equal_footing: bool = False,

    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    after_determinants: Union[int, Tuple[int, ...]] = 1,  #没有输入
    det_nlayer: Optional[int] = None,   #没有输入
    do_complex: bool = False,
    numb_k: int = 0,
    orb_numb_k: int = 0,
    do_twist: bool = False,
    do_aa: bool = False,
    det_mode: str = 'det',
    gemi_params: str = None,
    
) -> Tuple[InitFermiNet, FermiNetLike, FermiNetOptions]:
  """Creates functions for initializing parameters and evaluating ferminet.

  Args:
    atoms: (natom, ndim) array of atom positions.
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
    hf_solution: If present, initialise the parameters to match the Hartree-Fock
      solution. Otherwise a random initialisation is use.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    determinants: Number of determinants to use.
    after_determinants: currently ignored.

  Returns:
    init, network, options tuple, where init and network are callables which
    initialise the network parameters and apply the network respectively, and
    options specifies the settings used in the network.
  """
  del after_determinants,lattice
  assert (envelope and feature_layer and ferminet_model),("no envelope/feature_layer/ferminet_model")
  assert (det_mode == "det"),"only det mode is supported in gq"
  assert (numb_k==0 and orb_numb_k==0),"numb_k and orb_numb_k should be 0 in gq"
  gemi_ia=None

  options = FermiNetOptions(
      hidden_dims=hidden_dims,
      use_last_layer=use_last_layer,
      determinants=determinants,
      full_det=full_det,
      bias_orbitals=bias_orbitals,
      envelope=envelope,
      feature_layer=feature_layer,
      ferminet_model=ferminet_model,
      det_nlayer=det_nlayer,
      do_complex=do_complex,
      envelope_pw=None,  #None
      orb_env_pw=None,  #None
      do_twist=do_twist,
      do_aa=do_aa,
      mes=mes,
      equal_footing=equal_footing,
      det_mode=det_mode,
      gemi_params=gemi_params,
      gemi_ia=gemi_ia,
  )

  init = functools.partial(
      init_fermi_net_params,
      natom = natom, 
      ndim = ndim, 
      nspins=nspins,
      options=options,
      hf_solution=hf_solution,
  )
  network = functools.partial(
      fermi_net,
      nspins=nspins,
      options=options,
  )

  return init, network, options

