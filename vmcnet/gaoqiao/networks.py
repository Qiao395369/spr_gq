# Author:Qiaoqiao
# Date:2024/3/20
# filename:networks
# Description:
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
from typing import Any,Iterable,Mapping,Optional,Sequence,Tuple,Union

import attr
import chex
import vmcnet.gaoqiao.envelopes as envelopes
import vmcnet.gaoqiao.network_blocks as network_blocks
#import sto
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
import vmcnet.gaoqiao.dp as dp
#import attn,tri,gemi
import vmcnet.gaoqiao.gemi as gemi

FermiLayers=Tuple[Tuple[int,int],...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree=Union[jnp.ndarray,Iterable['ParamTree'],Mapping[Any,'ParamTree']]
#Union 类型表示 ParamTree 可以是括号内任何一个类型。
# 它是一种联合类型，用于表示一个值可以是几种不同类型中的任何一种。
#Iterable['ParamTree']: 表示参数树可以是任何可以迭代的对象，其元素本身也是 ParamTree 类型。
# 这允许参数树包含列表或元组等，其中每个元素也可能是一个复杂的参数树结构
#Mapping[Any, 'ParamTree']: 表示参数树可以是任何映射（如字典），它的值是 ParamTree 类型。
# 这使得参数树可以表示为键值对的集合，其中每个值可能是一个单独的参数、参数数组或更深层次的参数结构。

# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param=Mapping[str,jnp.ndarray]


## Interfaces (public) ##


class InitFermiNet(Protocol):

	def __call__(self,key: chex.PRNGKey) -> ParamTree:
		"""Returns initialized parameters for the network.

		Args:
		  key: RNG state
		"""


class FermiNetLike(Protocol):

	def __call__(self,params: ParamTree,
	             electrons: jnp.ndarray) -> Tuple[jnp.ndarray,jnp.ndarray]:
		"""Returns the sign and log magnitude of the wavefunction.

		Args:
		  params: network parameters.
		  electrons: electron positions, shape (nelectrons*ndim), where ndim is the
			dimensionality of the system.
		"""


class LogFermiNetLike(Protocol):

	def __call__(self,params: ParamTree,electrons: jnp.ndarray) -> jnp.ndarray:
		"""Returns the log magnitude of the wavefunction.

		Args:
		  params: network parameters.
		  electrons: electron positions, shape (nelectrons*ndim), where ndim is the
			dimensionality of the system.
		"""


## Interfaces (network components) ##


class FeatureInit(Protocol):

	def __call__(self) -> Tuple[Tuple[int,int],Param]:
		"""Creates the learnable parameters for the feature input layer.

		Returns:
		  Tuple of ((x, y), params), where x and y are the number of one-electron
		  features per electron and number of two-electron features per pair of
		  electrons respectively, and params is a (potentially empty) mapping of
		  learnable parameters associated with the feature construction layer.
		  (x,y)x是每个电子的单电子流特征数，y是每一对电子的双电子流特征数。
		"""


class FeatureApply(Protocol):

	def __call__(self,ae: jnp.ndarray,r_ae: jnp.ndarray,ee: jnp.ndarray,
	             r_ee: jnp.ndarray,
	             **params: jnp.ndarray) -> Tuple[jnp.ndarray,jnp.ndarray]:
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

	def __call__(self,key) -> Param:
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
	STANDARD=enum.auto()


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


@attr.s(auto_attribs=True,kw_only=True)
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
	ndim: int=3
	hidden_dims: FermiLayers=((256,32),(256,32),(256,32),(256,32))
	use_last_layer: bool=False
	determinants: int=16
	full_det: bool=True
	bias_orbitals: bool=False
	envelope: envelopes.Envelope=attr.ib(default=attr.Factory(envelopes.make_isotropic_envelope,takes_self=False))
	feature_layer: FeatureLayer=attr.ib(default=attr.Factory(lambda self:make_ferminet_features(ndim=self.ndim),takes_self=True))
	#当创建这个类的一个实例并且feature_layer属性没有显式赋值时，attr库将自动调用make_ferminet_features(ndim=self.ndim)
	#来生成一个默认的feature_layer值。这样做的好处是feature_layer属性的默认值可以根据实例的其他属性动态生成，增加了灵活性和可配置性。

	ferminet_model: FerminetModel=attr.ib(default=attr.Factory(lambda self:make_fermi_net_model,takes_self=True))
	det_nlayer: int=None
	do_complex: bool=False
	envelope_pw: Any=None
	orb_env_pw: Any=None
	do_aa: bool=False
	mes: dp.ManyElectronSystem=None
	det_mode: str='det'
	gemi_params: dict=None
	gemi_ia: Any=None
	equal_footing: bool=False


## Network initialisation ##

def init_layers(
		key: chex.PRNGKey,dims_one_in: Sequence[int],dims_one_out: Sequence[int],
		dims_two_in: Sequence[int],
		dims_two_out: Sequence[int]) -> Tuple[Sequence[Param],Sequence[Param]]:
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
	if len(dims_one_in)!=len(dims_one_out):
		raise ValueError(
			'Length of one-electron stream inputs and outputs not identical.')
	if len(dims_two_in)!=len(dims_two_out):
		raise ValueError(
			'Length of two-electron stream inputs and outputs not identical.')
	if len(dims_two_in) not in (len(dims_one_out),len(dims_one_out)-1):
		raise ValueError('Number of layers in two electron stream must match or be '
		                 'one fewer than the number of layers in the one-electron '
		                 'stream')
	single=[]
	double=[]
	ndouble_layers=len(dims_two_in)
	for i in range(len(dims_one_in)):
		key,subkey=jax.random.split(key)
		single.append(
			network_blocks.init_linear_layer(
				subkey,
				in_dim=dims_one_in[i],
				out_dim=dims_one_out[i],
				include_bias=True))

		if i<ndouble_layers:
			key,subkey=jax.random.split(key)
			double.append(
				network_blocks.init_linear_layer(
					subkey,
					in_dim=dims_two_in[i],
					out_dim=dims_two_out[i],
					include_bias=True))

	return single,double

#params['orbital']=init_orbital_shaping(
# key=subkey,
# input_dim=dims_orbital_in,  #16
# nspin_orbitals=nspin_orbitals_alloc, #[32,32]
# bias_orbitals=options.bias_orbitals  #False
#)
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
	orbitals=[]
	for nspin_orbital in nspin_orbitals:
		key,subkey=jax.random.split(key)
		orbitals.append(
			network_blocks.init_linear_layer(
				subkey,
				in_dim=input_dim,  #16
				out_dim=nspin_orbital,     #32
				include_bias=bias_orbitals))   #False
	return orbitals


def init_to_hf_solution(
		hf_solution,
		single_layers: Sequence[Param],
		orbital_layer: Sequence[Param],
		determinants: int,
		active_spin_channels: Sequence[int],
		eps: float = 0.01) -> Tuple[Sequence[Param],Sequence[Param]]:
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
	single_layers=jax.tree_map(lambda param:param*eps,single_layers)
	# Initialize first layer of Fermi Net to match s- or p-type orbitals.
	# The sto and sto-poly envelopes can exactly capture the s-type orbital,
	# so the effect of the neural network part is constant, while the p-type
	# orbital also has a term multiplied by x, y or z.
	j=0
	for ia,atom in enumerate(hf_solution.molecule):
		coeffs=sto.STO_6G_COEFFS[atom.symbol]
		for orb in coeffs.keys():
			if orb[1]=='s':
				single_layers[0]['b']=single_layers[0]['b'].at[j].set(1.0)
				j+=1
			elif orb[1]=='p':
				w=single_layers[0]['w']
				w=w.at[ia*4+1:(ia+1)*4,j:j+3].set(jnp.eye(3))
				single_layers[0]['w']=w
				j+=3
			else:
				raise NotImplementedError('HF Initialization not implemented for '
				                          f'{orb[1]} orbitals')
	# Scale all params in orbital shaping to be near zero.
	orbital_layer=jax.tree_map(lambda param:param*eps,orbital_layer)
	for i,spin in enumerate(active_spin_channels):
		# Initialize last layer to match Hartree-Fock weights on basis set.
		norb=hf_solution.mean_field.mo_coeff[i].shape[0]
		mat=hf_solution.mean_field.mo_coeff[i][:,:spin]
		w=orbital_layer[i]['w']
		for j in range(determinants):
			w=w.at[:norb,j*spin:(j+1)*spin].set(mat)
		orbital_layer[i]['w']=w
	return single_layers,orbital_layer


def init_fermi_net_params(
		key: chex.PRNGKey,
		natom: int,
		ndim: int,
		nspins: Tuple[int,...],
		options: FermiNetOptions,
		hf_solution=None,
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
	if options.envelope.apply_type==envelopes.EnvelopeType.PRE_ORBITAL:
		if options.bias_orbitals:
			raise ValueError('Cannot bias orbitals w/STO envelope.')
	if hf_solution is not None:
		if options.use_last_layer:
			raise ValueError('Cannot use last layer w/HF init')
		if options.envelope.apply_type not in ('sto','sto-poly'):
			raise ValueError('When using HF init, '
			                 'envelope_type must be `sto` or `sto-poly`.')

	active_spin_channels=[spin for spin in nspins if spin>0]
	nchannels=len(active_spin_channels)
	if nchannels==0:
		raise ValueError('No electrons present!')

	params={}
	key,subkey=jax.random.split(key,num=2)
	_params,dims_orbital_in=options.ferminet_model.init(subkey)
	 #网络中线性变换的参数
	for kk,vv in _params.items():
		params[kk]=vv

	# How many spin-orbitals do we need to create per spin channel?
	nspin_orbitals=[]
	if options.det_mode=="det":
		for nspin in active_spin_channels:
			if options.full_det:
				# Dense determinant. Need N orbitals per electron per determinant.
				norbitals=sum(nspins)*options.determinants
				  #(n1+n2)*nd
			else:
				# Spin-factored block-diagonal determinant. Need nspin orbitals per electron per determinant.
				norbitals=nspin*options.determinants
			nspin_orbitals.append(norbitals)
			#if full_det ,nspin_orbitals:[(n1+n2)*nd,(n1+n2)*nd]
			#if not full_fet,nspin_orbitals:[n1*nd,n2*nd]
	elif options.det_mode=="gemi":
		# assumes that nspins[0] >= nspins[1]
		diff_sp=nspins[0]-nspins[1]
		if nspins[1]==0:
			# case nspins = (spin, 0)
			nspin_orbitals.append(
				diff_sp*options.determinants)
		else:
			# case nspins = (spin0, spin1)
			nspin_orbitals.append(
				(options.gemi_params['odim']+diff_sp)*options.determinants)   #ptions.gemi_params['odim']=16，options.determinants=1
			nspin_orbitals.append(
				(options.gemi_params['odim'])*options.determinants)
			# print("nspin_orbitals:",nspin_orbitals)
			# 最终得到nspin_orbitals=[16,16]
	else:
		raise RuntimeError(f"unknown det_mode {options.det_mode}")

	if options.do_complex:
		# twice the output channels if do_complex
		nspin_orbitals_alloc=[ii*2 for ii in nspin_orbitals]  #[32,32]
	else:
		nspin_orbitals_alloc=nspin_orbitals

	# create envelope params
	if options.envelope.apply_type==envelopes.EnvelopeType.PRE_ORBITAL:
		# Applied to output from final layer of 1e stream.
		output_dims=dims_orbital_in
	elif options.envelope.apply_type==envelopes.EnvelopeType.PRE_DETERMINANT:
		# Applied to orbitals.
		output_dims=nspin_orbitals
	elif options.envelope.apply_type==envelopes.EnvelopeType.PRE_DETERMINANT_Z:
		# Applied to orbitals.
		output_dims=nspin_orbitals  #[16,16]
	elif options.envelope.apply_type==envelopes.EnvelopeType.PRE_DETERMINANT_TWIST:
		# Applied to orbitals.
		output_dims=nspin_orbitals
	elif options.envelope.apply_type==envelopes.EnvelopeType.POST_DETERMINANT:
		# Applied to all determinants.
		output_dims=1
	else:
		raise ValueError('Unknown envelope type')
	if options.envelope.apply_type!=envelopes.EnvelopeType.PRE_DETERMINANT_Z:
		params['envelope']=options.envelope.init(
			natom=natom,output_dims=output_dims,hf=hf_solution,ndim=ndim)
	else:
		params['envelope']=options.envelope.init(
			key,
			hz_size=options.hidden_dims[-1][0],   #(16)
			output_dims=output_dims     #[16,16]or[odim+diff,odim]
		)

	# orbital shaping
	key,subkey=jax.random.split(key,num=2)
	params['orbital']=init_orbital_shaping(
		key=subkey,
		input_dim=dims_orbital_in,  #16
		nspin_orbitals=nspin_orbitals_alloc, #[32,32]
		bias_orbitals=options.bias_orbitals)  #False
		#bias_orbitals=True) #对问题没用
	#即两个(16,32)(16,32)的变换矩阵
	#print('dims_orbital_in:',dims_orbital_in)
	#print('nspin_orbitals_alloc:',nspin_orbitals_alloc)
	if hf_solution is not None:     #hf_solution: None
		params['single'],params['orbital']=init_to_hf_solution(
			hf_solution=hf_solution,
			single_layers=params['single'],
			orbital_layer=params['orbital'],
			determinants=options.determinants,
			active_spin_channels=active_spin_channels,
			eps=eps)

	if options.det_nlayer is not None and options.det_nlayer>0:  #options.det_nlayer: None
		params['det']=[]
		for ii in range(options.det_nlayer):
			key,subkey=jax.random.split(key)
			params['det'].append(
				network_blocks.init_linear_layer(
					subkey,
					in_dim=options.determinants,
					out_dim=options.determinants,
					include_bias=False,
				))
	else:
		params['det']=None

	if options.det_mode=="gemi":
		key,subkey=jax.random.split(key)
		params['gemi']=options.gemi_ia[0](subkey)

	if options.envelope_pw is not None:   # options.envelope_pw: None
		key,subkey=jax.random.split(key)
		params['envelope_pw']=options.envelope_pw.init(
			key,dims_orbital_in,options.hidden_dims[-1][0])

	if options.orb_env_pw is not None:      # options.orb_env_pw: None
		key,subkey=jax.random.split(key)
		# backed by nspin_orbitals,
		# because output_dims is twice the nspin_orbitals if complex
		params['orb_env_pw']=options.orb_env_pw.init(
			key,dims_orbital_in,nspin_orbitals)

	return params


## Network layers ##


def construct_input_features(
		pos_: jnp.ndarray,
		atoms: jnp.ndarray,
		ndim: int = 3,
		do_aa: bool = False,
) -> Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray,jnp.ndarray]:
	"""Constructs inputs to Fermi Net from raw electron and atomic positions.

	Args:
	  pos: electron positions. Shape (nelectrons*ndim,).
	  atoms: atom positions. Shape (natoms, ndim).
	  ndim: dimension of system. Change only with caution.

	Returns:
	  ae, ee, r_ae, r_ee tuple, where:
		ae: atom-electron vector. Shape (nelectron, natom, ndim).
		ee: atom-electron vector. Shape (nelectron, nelectron, ndim).
		r_ae: atom-electron distance. Shape (nelectron, natom, 1).
		r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
	  The diagonal terms in r_ee are masked out such that the gradients of these
	  terms are also zero.
	"""
	pos=pos_
	# pos=jnp.reshape(pos_,(-1,3))
	# atoms=jnp.reshape(atoms,(-1,3))
	# print("pos.shape networks:",pos.shape)
	# print("atoms.shape networks:",atoms.shape)
	assert atoms.shape[1]==ndim
	ae=jnp.reshape(pos,[-1,1,ndim])-atoms[None,...]   #(nele,1,3)-(1,natoms,3)->(nele,natoms,3)
	ee=jnp.reshape(pos,[1,-1,ndim])-jnp.reshape(pos,[-1,1,ndim])  #(1,n,3)-(n,1,3)->(n,n,3)
	if do_aa:
		aa=jnp.reshape(atoms,[1,-1,ndim])-jnp.reshape(atoms,[-1,1,ndim])  #(1,n,3)-(n,1,3)->(n,n,3)
	else:
		aa=None

	r_ae=jnp.linalg.norm(ae,axis=2,keepdims=True)  #(ne,na,1)
	# Avoid computing the norm of zero, as is has undefined grad
	ne=ee.shape[0]
	r_ee=(jnp.linalg.norm(ee+jnp.eye(ne)[...,None],axis=-1)*(1.0-jnp.eye(ne)))[...,None] #(n,n,1)
	if do_aa:
		na=aa.shape[0]
		r_aa=(jnp.linalg.norm(aa+jnp.eye(na)[...,None],axis=-1)*(1.0-jnp.eye(na)))[...,None]
	else:
		r_aa=None

	return ae,ee,aa,r_ae,r_ee,r_aa

def make_fermi_net_model(
    atoms,
    nspins,
    feature_layer,
    hidden_dims,
    use_last_layer,
    dim_extra_params : int = 0,
    do_aa : bool=False,
    mes = None,
):
  assert(dim_extra_params == 0), "do not support extra input parameters!"
  del dim_extra_params, do_aa, mes
  natom, ndim = atoms.shape
  del atoms

  def init(
      subkey,
  ):
    params = {}
    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    (num_one_features, num_two_features), params['input'] = (feature_layer.init())
    # The input to layer L of the one-electron stream is from
    # construct_symmetric_features and shape (nelectrons, nfeatures), where
    # nfeatures is i) output from the previous one-electron layer; ii) the mean
    # for each spin channel from each layer; iii) the mean for each spin channel
    # from each two-electron layer. We don't create features for spin channels
    # which contain no electrons (i.e. spin-polarised systems).
    nfeatures = lambda out1, out2: (nchannels + 1) * out1 + nchannels * out2
    # one-electron stream, per electron:
    #  - one-electron features per atom (default: electron-atom vectors
    #    (ndim/atom) and distances (1/atom)),
    # two-electron stream, per pair of electrons:
    #  - two-electron features per electron pair (default: electron-electron
    #    vector (dim) and distance (1))
    feature_one_dims = natom * num_one_features
    feature_two_dims = num_two_features
    dims_one_in = (
        [nfeatures(feature_one_dims, feature_two_dims)] +
        [nfeatures(hdim[0], hdim[1]) for hdim in hidden_dims[:-1]])
    dims_one_out = [hdim[0] for hdim in hidden_dims]
    if use_last_layer:
      dims_two_in = ([feature_two_dims] +
                     [hdim[1] for hdim in hidden_dims[:-1]])
      dims_two_out = [hdim[1] for hdim in hidden_dims]
    else:
      dims_two_in = ([feature_two_dims] +
                     [hdim[1] for hdim in hidden_dims[:-2]])
      dims_two_out = [hdim[1] for hdim in hidden_dims[:-1]]
    # Layer initialisation
    params['single'], params['double'] = init_layers(
        key=subkey,
        dims_one_in=dims_one_in,
        dims_one_out=dims_one_out,
        dims_two_in=dims_two_in,
        dims_two_out=dims_two_out)
    # orbital input dims
    if not use_last_layer:
      # Just pass the activations from the final layer of the one-electron stream
      # directly to orbital shaping.
      dims_orbital_in = hidden_dims[-1][0]
    else:
      dims_orbital_in = nfeatures(hidden_dims[-1][0],
                                  hidden_dims[-1][1])
    return params, dims_orbital_in

  def apply(
      params,
      ae_features,
      ee_features,
      aa_features=None,
      pos=None,
      lattice=None,
      extra_params=None,
  ):
    del pos, lattice
    h_one = ae_features
    h_two = ee_features
    residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
    for i in range(len(params['double'])):
      h_one_in = construct_symmetric_features(h_one, h_two, nspins)

      # Execute next layer
      h_one_next = jnp.tanh(
          network_blocks.linear_layer(h_one_in, **params['single'][i]))
      h_two_next = jnp.tanh(
          network_blocks.vmap_linear_layer(h_two, params['double'][i]['w'],
                                           params['double'][i]['b']))
      h_one = residual(h_one, h_one_next)
      h_two = residual(h_two, h_two_next)
    if len(params['double']) != len(params['single']):
      h_one_in = construct_symmetric_features(h_one, h_two, nspins)
      h_one_next = jnp.tanh(
          network_blocks.linear_layer(h_one_in, **params['single'][-1]))
      h_one = residual(h_one, h_one_next)
      h_to_orbitals = h_one
    else:
      h_to_orbitals = construct_symmetric_features(h_one, h_two, nspins)
    return h_to_orbitals, None

  return FerminetModel(init, apply)
def make_ferminet_features(charges: Optional[jnp.ndarray] = None,
                           nspins: Optional[Tuple[int, ...]] = None,
                           ndim: int = 3) -> FeatureLayer:
  """Returns the init and apply functions for the standard features."""

  del charges, nspins

  def init() -> Tuple[Tuple[int, int], Param]:
    return (ndim + 1, ndim + 1), {}

  def apply(ae, r_ae, ee, r_ee, aa=None, r_aa=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    do_aa = aa is not None and r_aa is not None
    ae_features = jnp.concatenate((r_ae, ae), axis=2)
    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    ee_features = jnp.concatenate((r_ee, ee), axis=2)
    if do_aa:
      aa_features = jnp.concatenate((r_aa, aa), axis=2)
    else:
      aa_features = None
    return ae_features, ee_features, aa_features

  return FeatureLayer(init=init, apply=apply)

def make_fermi_net_model_zinv_shrd(
		natom,
		ndim,
		nspins,
		feature_layer,
		hidden_dims,
		use_last_layer,
		dim_extra_params,   #0
		do_aa: bool = True,
		mes=None,
		# extra parameters
		distinguish_ele: bool = False,
		code_only_first: bool = False,
		attn_params: Optional[dict] = None,
		attn_1_params: Optional[dict] = None,
):
	"""
   natom: 14
   ndim: 3
   nspins: (7, 7)
   hidden_dims: ((16, 16), (16, 16), (16, 16), (16, 16))
   use_last_layer: False
   dim_extra_params: 0
   do_aa: True
   distinguish_ele: True
   code_only_first: True
   """
	do_attn=attn_params is not None
	do_attn_1=attn_1_params is not None
	#do_atten=False
	#do_atten_1=False
	if do_attn:
		attn_qkdim=attn_params.get("qkdim",8)
		attn_nhead=attn_params.get("nhead",2)

	if not do_aa:
		raise RuntimeError("do_aa must be true when using shared parameters")
	if mes is None:
		raise RuntimeError("mes should not be None when using shared parameters")
	if do_attn and do_attn_1:
		raise RuntimeError("cannot do both attention mechanisms in the same model")

	def init(key,):
		dim_posi_code=mes.get_dim_one_hot()  #3
		dim_pair_code=2*mes.get_dim_one_hot()  #6
		# extra params
		dim_1_append=dim_extra_params+dim_posi_code  
		#dim_extra_params:r,w....
		dim_2_append=dim_extra_params+dim_pair_code  
		# number of spin channel
		active_spin_channels=[spin for spin in nspins if spin>0]
		nchannels=len(active_spin_channels)  #2
		# init params
		params={}
		(num_one_features,num_two_features),params['input']=(feature_layer.init())
		if not distinguish_ele:
			nfeatures=lambda out1,out2:2*out1+2*out2
		else:
			nfeatures=lambda out1,out2:(nchannels+1)*out1+(nchannels+1)*out2
		dims_1_in=[nfeatures(num_two_features+dim_1_append,num_two_features)]
		if code_only_first:
			dims_1_in+=[nfeatures(hdim[0],hdim[1]) for hdim in hidden_dims[:-1]]
		else:
			dims_1_in+=[nfeatures(hdim[0]+dim_1_append,hdim[1]) for hdim in hidden_dims[:-1]]
		dims_1_out=[hdim[0] for hdim in hidden_dims]
		# dims_1_in = [ii + dim_1_append for ii in dims_1_in]
		if use_last_layer:
			dims_2_in=([num_two_features]+
			           [hdim[1] for hdim in hidden_dims[:-1]])
			dims_2_out=[hdim[1] for hdim in hidden_dims]
		else:
			dims_2_in=([num_two_features]+
			           [hdim[1] for hdim in hidden_dims[:-2]])
			dims_2_out=[hdim[1] for hdim in hidden_dims[:-1]]
		if code_only_first:
			dims_2_in[0]+=dim_2_append
		else:
			dims_2_in=[ii+dim_2_append for ii in dims_2_in]
		#print('hidden_dims:',hidden_dims,'\n',
		    #   'dim_1_append:',dim_1_append,'\n',
		    #  'dim_2_append:',dim_2_append,'\n',
		    #  'dims_1_in:',dims_1_in,'\n',
		    #  'dims_1_out:',dims_1_out,'\n',
		    #  'dims_2_in:',dims_2_in,'\n',
		     # 'dims_2_out:',dims_2_out,)

#depth=4,h1=h2=16
		#hidden_dims: ((16,16),(16,16),(16,16),(16,16))
		#dim_1_append: 3
		#dim_2_append: 6
		#dims_1_in: [33,96,96,96]
		#dims_1_out: [16,16,16,16]
		#dims_2_in: [10,16,16]
		#dims_2_out: [16,16,16]

#depth=4,h1=h2=8
		#hidden_dims: ((8,8),(8,8),(8,8),(8,8))
		#dim_1_append: 3
		#dim_2_append: 6
		#dims_1_in: [33,48,48,48]
		#dims_1_out: [8,8,8,8]
		#dims_2_in: [10,8,8]
		#dims_2_out: [8,8,8]

#depth=4,h1=h2=4
		#hidden_dims: ((4,4),(4,4),(4,4),(4,4))
		#dim_1_append: 3
		#dim_2_append: 6
		#dims_1_in: [33,24,24,24]
		#dims_1_out: [4,4,4,4]
		#dims_2_in: [10,4,4]
		#dims_2_out: [4,4,4]




#hidden_dims: ((16, 16), (16, 16), (16, 16), (16, 16))
		#dim_1_append: 6
		#dim_2_append: 9
		#dims_1_in: [60, 96, 96, 96]
		#dims_1_out: [16, 16, 16, 16]
		#dims_2_in: [16, 16, 16]
		#dims_2_out: [16, 16, 16]

		# Layer initialisation
		key,subkey=jax.random.split(key)
		params['one'],params['two']=init_layers(
			key=subkey,
			dims_one_in=dims_1_in,
			dims_one_out=dims_1_out,
			dims_two_in=dims_2_in,
			dims_two_out=dims_2_out)
		if not use_last_layer:
			# Just pass the activations from the final layer of the one-electron stream
			# directly to orbital shaping.
			dims_orbital_in=hidden_dims[-1][0]  #16
		else:
			dims_orbital_in=nfeatures(hidden_dims[-1][0],hidden_dims[-1][1])
		# projection parameters
		dim_proj_1_in=dims_1_out[:len(dims_2_out)]
		dim_proj_1_out=dims_2_out
		if not code_only_first:
			dim_proj_1_in=[ii+dim_1_append for ii in dim_proj_1_in]
		params['proj']=[]
		for ii in range(len(params['two'])):
			if dim_proj_1_in[ii]!=dim_proj_1_out[ii]:
				key,subkey=jax.random.split(key)
				params['proj'].append(network_blocks.init_linear_layer(subkey,dim_proj_1_in[ii],dim_proj_1_out[ii],))
			else:
				# do not project if the input and output dims are the same
				params['proj'].append(None)
		dim_proj_0_in=num_one_features+dim_1_append
		dim_proj_0_out=num_two_features
		if dim_proj_0_in!=dim_proj_0_out:
			key,subkey=jax.random.split(key)
			params['proj_0']=network_blocks.init_linear_layer(
				subkey,dim_proj_0_in,dim_proj_0_out,)
		else:
			params['proj_0']=None
		# print("dim_proj_0_in:",dim_proj_0_in)   #7
		#dim_proj_1_in: [16, 16, 16]
		#dim_proj_1_out: [16, 16, 16]
		# dim_proj_0_in: 13
		# dim_proj_0_out: 7

		# attn
		if do_attn:
			dim_attn_in=[dim_proj_0_in]+dim_proj_1_in
			dim_attn_out=[attn_qkdim]*len(dim_attn_in)
			dim_attn_out=[ii*attn_nhead for ii in dim_attn_out]
			params['attn_qmap']=[]
			params['attn_kmap']=[]
			params['attn_headw']=[]
			for ii in range(len(dim_attn_in)):
				key,subkeyq,subkeyk=jax.random.split(key,3)
				params['attn_qmap'].append(network_blocks.init_linear_layer(
					subkeyq,dim_attn_in[ii],dim_attn_out[ii],include_bias=False))
				params['attn_kmap'].append(network_blocks.init_linear_layer(
					subkeyk,dim_attn_in[ii],dim_attn_out[ii],include_bias=False))
				key,subkey=jax.random.split(key)
				params['attn_headw'].append(network_blocks.init_linear_layer(
					subkey,attn_nhead,1,include_bias=False))
		return params,dims_orbital_in

	def _1_apply(
			he,
			params,
			activation: bool = True
	):
		ret=network_blocks.linear_layer(he,**params)
		if activation:
			ret=jnp.tanh(ret)
		return ret

	def collective_1_apply(
			he,hi,
			params,
			activation: bool = True
	):
		nelecs=he.shape[0]
		ret=network_blocks.linear_layer(jnp.concatenate([he,hi],axis=0),**params)
		if activation:
			ret=jnp.tanh(ret)
		ret=jnp.split(ret,[nelecs],axis=0)
		return ret[0],ret[1]

	def collective_2_apply(
			hee,hei,hii,
			params,
			hie: Optional[jnp.ndarray] = None,
			activation: bool = True,
	):
		do_hie=hie is not None
		nelecs=hee.shape[0]
		natoms=hii.shape[0]
		nfeats=hee.shape[-1]
		clist=[
			hee.reshape([-1,nfeats]),
			hei.reshape([-1,nfeats]),
			hii.reshape([-1,nfeats]),
		]
		if do_hie:
			clist.append(hie.reshape([-1,nfeats]))
		xx=jnp.concatenate(clist,axis=0)
		ret=network_blocks.linear_layer(xx,**params)
		if activation:
			ret=jnp.tanh(ret)
		nfeats=ret.shape[-1]
		slist=[nelecs*nelecs,nelecs*(nelecs+natoms)]
		if do_hie:
			slist.append(nelecs*(nelecs+natoms)+natoms*natoms)
		ret=jnp.split(ret,slist,axis=0)
		ret[0]=ret[0].reshape([nelecs,nelecs,nfeats])
		ret[1]=ret[1].reshape([nelecs,natoms,nfeats])
		ret[2]=ret[2].reshape([natoms,natoms,nfeats])
		if do_hie:
			ret[3]=ret[3].reshape([natoms,nelecs,nfeats])
		else:
			ret.append(None)
		return ret[0],ret[1],ret[2],ret[3]

	def attention_map(qq,kk,axis=1):
		_,dd,nh=qq.shape
		return jax.nn.softmax(
			jnp.einsum("ikh,jkh->ijh",qq,kk)/jnp.sqrt(dd),
			axis=axis,
		)

	def collective_attn_head_apply(list_data,params):
		list_shape=[ii.shape for ii in list_data]
		nh=list_shape[0][-1]
		list_data=[ii.reshape(-1,nh) for ii in list_data]
		list_data_split=np.cumsum([ii.shape[0] for ii in list_data])[:-1]
		coll_data=jnp.concatenate(list_data,axis=0)
		coll_data=network_blocks.linear_layer(coll_data,**params).reshape(-1)
		list_data=jnp.split(coll_data,list_data_split)
		list_data=[ii.reshape(ss[:-1]) for ii,ss in zip(list_data,list_shape)]
		return list_data

	def construct_symmetric_features_conv(
			hz: jnp.ndarray,
			hi: jnp.ndarray,
			hiz: jnp.ndarray,
			hij: jnp.ndarray,
			hyz: jnp.ndarray,
			spins: Tuple[int,int],
			hzi: Optional[jnp.ndarray] = None,
			proj: Optional[Mapping[str,jnp.ndarray]] = None,
			attn_params: Optional[Tuple[Mapping[str,jnp.ndarray]]] = None,
	) -> jnp.ndarray:
		"""
		hz  : nz x nfiz  (n,13)
		hi  : nele x nfij  (n,13)
		hiz : nele x nz x nfiz  (n,n,7)
		hij : nele x nele x nfij  (n,n,7)
		hyz : nz x nz x nf  (n,n,7)
		"""
		has_hzi=hzi is not None  #hzi=None
		if proj is not None:
			phi,phz=collective_1_apply(hi,hz,proj,activation=False)
		#i==0时，将hi和hz叠成(2n,13)，这2n行用同一个线性变换，变成(2n,7)，再分开为(n,,7),(n,7)。
		#i!=0时，proj is None。
		else:
			phi,phz=hi,hz
		# nele x nz x nf
		# print("pho.shape:",phi.shape)
		hizxhi=hiz*phi[:,None,:]  #(n,n,7)*(n,1,7)->(n,n,7)
		if not has_hzi:
			hizxhz=hiz*phz[None,:,:]  ##(n,n,7)*(1,n,7)->(n,n,7)
		else:
			hizxhz=jnp.transpose(hzi*phz[:,None,:],(1,0,2))
		# nele x nele x nf
		hijxhi=hij*phi[:,None,:]  #(n,n,7)*(n,1,7)->(n,n,7)
		# nz x nz x nf
		hyzxhz=hyz*phz[:,None,:]  #(n,n,7)*(n,1,7)->(n,n,7)
		if do_attn:
			q_map,k_map,head_w=attn_params
			qhi,qhz=collective_1_apply(hi,hz,q_map,activation=False)
			khi,khz=collective_1_apply(hi,hz,k_map,activation=False)
			[qhi,qhz,khi,khz]=[
				ii.reshape(-1,attn_qkdim,attn_nhead) for ii in\
				[qhi,qhz,khi,khz]]
			# nele x nz x nh
			amap_zi=attention_map(khi,qhz,axis=0)
			# nele x nz x nh
			amap_iz=attention_map(qhi,khz,axis=1)
			# nele x nele x nh
			amap_ii=attention_map(khi,qhi,axis=0)
			# nz x nz x nh
			amap_zz=attention_map(khz,qhz,axis=0)
			# update with attention maps
			hizxhi=hizxhi[:,:,:,None]*amap_zi[:,:,None,:]
			hizxhz=hizxhz[:,:,:,None]*amap_iz[:,:,None,:]
			hijxhi=hijxhi[:,:,:,None]*amap_ii[:,:,None,:]
			hyzxhz=hyzxhz[:,:,:,None]*amap_zz[:,:,None,:]
			[hizxhi,hizxhz,hijxhi,hyzxhz]=collective_attn_head_apply(
				[hizxhi,hizxhz,hijxhi,hyzxhz],head_w)
		if not distinguish_ele:
			# [nz x nfiz]
			hiz_z=[jnp.mean(hizxhi,axis=0)]
			# [nele x nfiz]
			hiz_i=[jnp.mean(hizxhz,axis=1)]
			# [nele x nfij]
			hij_i=[jnp.mean(hijxhi,axis=0)]
			# [nz x nfyz]
			hyz_z=[jnp.mean(hyzxhz,axis=0)]
			# 1 x nfiz
			gz=jnp.mean(hz,axis=0,keepdims=1)
			# nz x nfiz
			gz=[jnp.tile(gz,[hz.shape[0],1])]
			# 1 x nfij
			gi=jnp.mean(hi,axis=0,keepdims=1)
			# nele x nfij
			gi=[jnp.tile(gi,[hi.shape[0],1])]
		else:
			spin_partitions=network_blocks.array_partitions(nspins)  #[n//2]即[7]
			# [nz x nfiz, nz x nfiz]
			hiz_z=[jnp.mean(h,axis=0) for h in jnp.split(hizxhi,spin_partitions,axis=0) if h.size>0]  #(14,14,7)->[(7,14,7),(7,14,7)]->[(14,7),(14,7)]即gI后三项中的前两项
			# nele x nfiz
			hiz_i=[jnp.mean(hizxhz,axis=1)]  #(14,14,7)->(14,7)
			# [nele x nfij, nele x nfij]
			hij_i=[jnp.mean(h,axis=0) for h in jnp.split(hijxhi,spin_partitions,axis=0) if h.size>0]  #(14,14,7)->[(7,14,7),(7,14,7)]->[(14,7),(14,7)]即gi后三项中的前两项
			# [nz x nfyz]
			hyz_z=[jnp.mean(hyzxhz,axis=0)] if do_aa else []  #(14,7)
			# 1 x nfiz
			gz=jnp.mean(hz,axis=0,keepdims=1)  #(1,13)
			# nz x nfiz
			gz=jnp.tile(gz,[hz.shape[0],1])  #(14,13)
			# [nz x nfiz, nz x nfiz]
			gz=[gz,gz]  #[(14,13),(14,13)]  即gI的前三项的后两项
			# [1 x nfij, 1 x nfij]
			gi=[jnp.mean(h,axis=0,keepdims=1) for h in jnp.split(hi,spin_partitions,axis=0) if
			    h.size>0]  #(14,13)->[(7,13),(7,13)]->[(1,13),(1,13)]
			# [nele x nfij, nele x nfij]
			gi=[jnp.tile(g,[hi.shape[0],1]) for g in gi]  #[(14,13),(14,13)]即gi前三项的后两项。
		return jnp.concatenate([hz]+gz+hiz_z+hyz_z,axis=-1),jnp.concatenate([hi]+gi+hij_i+hiz_i,axis=-1)

	#i==0时为(14,60)，(14,60)
	#i!=0时为(14,96)，(14,96)

	#options.ferminet_model.apply(params, hae, hee, aa_features=haa, pos=pos, lattice=options.lattice, extra_params=extra_params)
	def apply(
			params,
			ae_features,
			ee_features,
			aa_features=None,
			pos=None,
			extra_params=None,
	):
		part_one_hot=mes.get_part_one_hot()  #(2n,3)
		pair_one_hot=mes.get_pair_one_hot()  #(2n,2n,6)
		[ci,cz]=mes.split_ea(part_one_hot)  #[(n,3)电子,(n,3)质子]
		[cij,ciz,czi,cyz]=mes.split_ee_ea_ae_aa(pair_one_hot)  #[(n,n,6),(n,n,6),(n,n,6),(n,n,6)]
		# print("part_one_hot.shape:",part_one_hot.shape,'\n',
		#       "pair_one_hot.shape:",pair_one_hot.shape)
		do_ep=dim_extra_params>0  #False
		nelecs=mes.nelecs  #n
		natoms=mes.natoms  #n
		#extra_params=twist:(3,)
		ei=jnp.tile(extra_params[None,:],[nelecs,1]) if do_ep else None  #None
		ez=jnp.tile(extra_params[None,:],[natoms,1]) if do_ep else None
		eij=jnp.tile(extra_params[None,None,:],[nelecs,nelecs,1]) if do_ep else None  #None
		eiz=jnp.tile(extra_params[None,None,:],[nelecs,natoms,1]) if do_ep else None  #None
		ezi=jnp.tile(extra_params[None,None,:],[natoms,nelecs,1]) if do_ep else None  #None
		eyz=jnp.tile(extra_params[None,None,:],[natoms,natoms,1]) if do_ep else None
		cond_concat=lambda ci,ei:ci if ei is None else jnp.concatenate([ei,ci],axis=-1)
		ai=cond_concat(ci,ei)  #(n,3)
		az=cond_concat(cz,ez)  #(n,3)
		aij=cond_concat(cij,eij)  #(n,n,6)
		aiz=cond_concat(ciz,eiz)  #(n,n,6)
		azi=cond_concat(czi,ezi)  #(n,n,6)
		ayz=cond_concat(cyz,eyz)  #(n,n,6)

		if dim_extra_params>0:
			assert (extra_params is not None),"should provide extra parameters!"
			assert (extra_params.size==dim_extra_params),"the size of the extra parameter should be "+str(
				dim_extra_params)
		else:
			assert (extra_params is None),"extra parameters should be None because dim_extra_params was set to 0"

		residual=lambda x,y:(x+y)/jnp.sqrt(2.0) if x.shape==y.shape else y
		hiz=ae_features  #(n,n,4)
		hij=ee_features  #(n,n,4)
		hyz=aa_features  #(n,n,4)
		if do_attn_1:  #False
			hzi=jnp.transpose(hiz,(1,0,2))
		else:
			hzi=None
		hz=jnp.mean(hyz,axis=0)  #(n,4)
		hi=jnp.mean(hij,axis=0)  #(n,4)
		for i in range(len(params['two'])):
			if i==0 or not code_only_first:  #code_only_first=True
				[hi,hz]=[jnp.concatenate([ii,jj],axis=-1) for ii,jj in zip([hi,hz],[ai,az])]
			#zip函数将两个列表中相对应位置的元素打包成一个个元组，所以zip([hi, hz], [ai, az])会产生一个迭代器，其元素为(hi, ai)和(hz, az)。
			#最终hi=jnp.concatenate([hi,ai], axis=-1),hz=jnp.concatenate([hz,az], axis=-1)
			#hi:(n,4+3),hz:(n,4+3)

			# update channel one feature
			if do_attn:
				attn_params=(params['attn_qmap'][i],params['attn_kmap'][i],params['attn_headw'][i])
			else:
				attn_params=None
			if i==0:
				hz_in,hi_in=construct_symmetric_features_conv(
					hz,hi,hiz,hij,hyz,nspins,
					hzi=hzi,
					proj=params['proj_0'],
					attn_params=attn_params,
				)
			else:
				hz_in,hi_in=construct_symmetric_features_conv(
					hz,hi,hiz,hij,hyz,nspins,
					hzi=hzi,
					proj=params['proj'][i-1],
					attn_params=attn_params,
				)
			if i==0 or not code_only_first:
				[hij,hiz,hyz]=[jnp.concatenate([ii,jj],axis=-1) for ii,jj in zip([hij,hiz,hyz],[aij,aiz,ayz])]
				#(n,n,4+6)
				if do_attn_1:
					hzi=jnp.concatenate([hzi,azi],axis=-1)
			# channel one
			hi_next,hz_next=collective_1_apply(hi_in,hz_in,params['one'][i])
			# channel two
			hij_next,hiz_next,hyz_next,hzi_next=collective_2_apply(hij,hiz,hyz,params['two'][i],hie=hzi)
			hz=residual(hz,hz_next)
			hi=residual(hi,hi_next)
			hiz=residual(hiz,hiz_next)
			hij=residual(hij,hij_next)
			hyz=residual(hyz,hyz_next)
			if do_attn_1:
				hzi=residual(hzi,hzi_next)
		if len(params['two'])!=len(params['one']):
			if not code_only_first:
				[hi,hz]=[jnp.concatenate([ii,jj],axis=-1) for ii,jj in zip([hi,hz],[ai,az])]
			if do_attn:
				attn_params=(params['attn_qmap'][-1],params['attn_kmap'][-1],params['attn_headw'][-1])
			else:
				attn_params=None
			hz_in,hi_in=construct_symmetric_features_conv(hz,hi,hiz,hij,hyz,nspins,hzi=hzi,proj=params['proj'][-1], attn_params=attn_params,)
			#hz_in:(14,96), hi_in:(14,96)
			# channel one
			hi_next,hz_next=collective_1_apply(hi_in,hz_in,params['one'][-1])  #hi_next:(14,16), hz_next:(14,16)
			hi=residual(hi,hi_next)  #(14,16)
			hz=residual(hz,hz_next)  #(14,16)
			h_to_orbitals=hi
		else:
			_,h_to_orbitals=construct_symmetric_features_conv(hz,hi,hiz,hij,hyz,nspins,params['proj_z'][-1],params['proj_i'][-1])
		return h_to_orbitals,hz

	return FerminetModel(init,apply)


def fermi_net_orbitals(
		params,
		pos: jnp.ndarray,
		atoms: jnp.ndarray,
		nspins: Tuple[int,...],
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

	  当options.full_det为True时：你正在使用一个形状为(ndet, nalpha+nbeta, nalpha+nbeta)的单个矩阵。
	  这里，ndet代表行列式的数量，nalpha代表α电子的数量，nbeta代表β电子的数量。这个矩阵在输入交换下进行列交换，
	  这在量子化学模拟中是常见的操作，特别是在处理Slater行列式和电子相关问题时。
	  当options.full_det为False时：不是使用单个矩阵，而是有两个分开的矩阵：
	  一个形状为(ndet, nalpha, nalpha)的矩阵，用于α电子。另一个形状为(ndet, nbeta, nbeta)的矩阵，用于β电子。
	  这样的设计允许模拟分别处理α电子和β电子的行为，这在考虑电子自旋时非常重要。
	"""

	h_to_orbitals,hz,pairs=fermi_net_orbitals_part1(params,pos,atoms,nspins,options)
	ret=fermi_net_orbitals_part2(params,pos,h_to_orbitals,hz,pairs,nspins,options)
	return ret


def fermi_net_orbitals_part1(
		params,
		pos: jnp.ndarray,
		atoms: jnp.ndarray,
		nspins: Tuple[int,...],
		options: FermiNetOptions = FermiNetOptions(),
):
	ae,ee,aa,r_ae,r_ee,r_aa=construct_input_features(pos,atoms,do_aa=options.do_aa)
	#print('ae:',ae)
	#print('atoms:',atoms)
	#print('aa.shape:',aa.shape)
	#print('ae.shape:',ae.shape)
	#print('ee.shape:',ee.shape)
	#print('r_aa.shape:',r_aa.shape)
	#print('r_ae.shape:',r_ae.shape)
	#print('r_ee.shape:',r_ee.shape)
	#print('aa:',aa)
	#print('ae:',ae)
	ae_features,ee_features,aa_features=options.feature_layer.apply(ae=ae,r_ae=r_ae,ee=ee,r_ee=r_ee,aa=aa,r_aa=r_aa,**params['input'])
	#print('aa_features.shape:',aa_features.shape)
	#print('ae_features.shape:',ae_features.shape)
	#print('ee_features.shape:',ee_features.shape)
	#print('aa_features:',aa_features)
	hae=ae_features  # electron-ion features
	hee=ee_features  # two-electron features
	haa=aa_features  # two-ion features
	model_h_to_orbitals,hz=options.ferminet_model.apply(params,hae,hee,aa_features=haa,pos=pos)
	h_to_orbitals=model_h_to_orbitals
	#print('h_to_orbitals:',h_to_orbitals[0],'\n','hz:',hz[0])

	return h_to_orbitals,hz,(ee,ae,aa,r_ee,r_ae,r_aa)

def fermi_net_orbitals_part2(
		params,
		pos,
		h_to_orbitals,
		hz,
		pair_dist,
		nspins: Tuple[int,...],
		options: FermiNetOptions = FermiNetOptions(),
		extra_params: jnp.ndarray = None,
):
	model_h_to_orbitals=h_to_orbitals #(14,16)
	(ee,ae,aa,r_ee,r_ae,r_aa)=pair_dist

	#ptions.envelope.apply_type == EnvelopeType.PRE_DETERMINANT_Z
	if options.envelope.apply_type==envelopes.EnvelopeType.PRE_ORBITAL:
		envelope_factor=options.envelope.apply(ae=ae,r_ae=r_ae,r_ee=r_ee,**params['envelope'])
		h_to_orbitals=envelope_factor*h_to_orbitals
	# Note split creates arrays of size 0 for spin channels without any electrons.
	h_to_orbitals=jnp.split(h_to_orbitals,network_blocks.array_partitions(nspins),axis=0)
	#(7,16),(7,16)
	# Drop unoccupied spin channels
	h_to_orbitals=[h for h,spin in zip(h_to_orbitals,nspins) if spin>0]
	active_spin_channels=[spin for spin in nspins if spin>0]
	active_spin_partitions=network_blocks.array_partitions(active_spin_channels)
	# Create orbitals.
	orbitals=[network_blocks.linear_layer(h,**p) for h,p in zip(h_to_orbitals,params['orbital'])]
	#得到(7,32)(7,32)
	# orbitals[0]=jnp.tanh(orbitals[0])
	# orbitals[1]=jnp.tanh(orbitals[1])

	# if do_complex, make the orbitals complex numbers
	if options.do_complex:
		for ii in range(len(active_spin_channels)):
			nsplit=orbitals[ii].shape[-1]//2  #16
			split_orb=jnp.split(orbitals[ii],[nsplit],axis=-1)  #(7,16)(7,16)
			orbitals[ii]=split_orb[0]+1j*split_orb[1]    #(7,16)
	#orbital:[(7,16),(7,16)]

	# Apply envelopes if required.
	if options.envelope.apply_type==envelopes.EnvelopeType.PRE_DETERMINANT:
		ae_channels=jnp.split(ae,active_spin_partitions,axis=0)
		r_ae_channels=jnp.split(r_ae,active_spin_partitions,axis=0)
		r_ee_channels=jnp.split(r_ee,active_spin_partitions,axis=0)
		for i in range(len(active_spin_channels)):
			orbitals[i]=orbitals[i]*options.envelope.apply(
				ae=ae_channels[i],
				r_ae=r_ae_channels[i],
				r_ee=r_ee_channels[i],
				**params['envelope'][i],
			)
	elif options.envelope.apply_type==envelopes.EnvelopeType.PRE_DETERMINANT_Z:
		ae_channels=jnp.split(ae,active_spin_partitions,axis=0)
		#ae:(14,14,3)->(7,14,3)
		for i in range(len(active_spin_channels)):
			orbitals[i]=orbitals[i]*options.envelope.apply(   #这一步就是算法第5行，对每一个orbital元素乘上一个envelope项
				hz=hz,    #hz:(14,16)质子单流
				ae=ae_channels[i], #(7,14,3)
				**params['envelope'][i],
			)
		#orbitals:[(7,16),(7,16)]
	#这里已经把两种自旋分开来了，就是第6行右边的Phi_{i,alpha}和Phi_{j,alpha},alpha为特征维，即16那一维。
	#这里的16是其实是16*K，如果有两个行列式，则K=2，那么这里alpha维应该是32，后续被reshape成两个16
		#print('orbitals:',orbitals[0][0][0],'\n','orbitals.shape:',orbitals[0].shape)
	#上面说的K就是ndet，即nd

	elif options.envelope.apply_type==envelopes.EnvelopeType.PRE_DETERMINANT_TWIST:
		assert (options.do_twist),"should do twist to use envelope of type PRE_DETERMINANT_TWIST"
		ae_channels=jnp.split(ae,active_spin_partitions,axis=0)
		r_ae_channels=jnp.split(r_ae,active_spin_partitions,axis=0)
		r_ee_channels=jnp.split(r_ee,active_spin_partitions,axis=0)
		for i in range(len(active_spin_channels)):
			orbitals[i]=orbitals[i]*options.envelope.apply(
				ae=ae_channels[i],
				r_ae=r_ae_channels[i],
				r_ee=r_ee_channels[i],
				**params['envelope'][i],
				twist=extra_params,
			)

	# Apply orbital planewave envelope
	if options.orb_env_pw is not None:  #options.orb_env_pw: None
		pos_channels=jnp.split(pos.reshape([-1,3]),active_spin_partitions,axis=0)
		h_channels=jnp.split(model_h_to_orbitals,active_spin_partitions,axis=0)
		for ii in range(len(active_spin_channels)):
			orbitals[ii]=orbitals[ii]*options.orb_env_pw.apply(
				pos=pos_channels[ii],
				he=h_channels[ii],
				twist=extra_params,
				**params['orb_env_pw'][ii],
			)

	#options.det_mode: gemi
	# Reshape into matrices.
	if options.det_mode=="det":
		shapes=[(spin,-1,sum(nspins) if options.full_det else spin)
		        for spin in active_spin_channels]
		#shape:[(n1,-1,n1),(n2,-1,n2)]
	else:
		shapes=[(spin,options.determinants,-1)
		        for spin in active_spin_channels]
		#shape:[(7,1,-1),(7,1,-1)]
	orbitals=[jnp.reshape(orbital,shape) for orbital,shape in zip(orbitals,shapes)]
	#orbitals:[(7,1,16),(7,1,16)]
	orbitals=[jnp.transpose(orbital,(1,0,2)) for orbital in orbitals]
	#orbitals:[(1,7,16),(1,7,16)]
	# print("full_det:",options.full_det)
	if options.full_det:
		orbitals=[jnp.concatenate(orbitals,axis=1)]

	return orbitals,(ae,r_ae,r_ee,model_h_to_orbitals,hz)
#这里的orbitals就是gemi中的xs,vmap_loggemi/loggemi中的x0和x1。


## FermiNet ##



def fermi_net(
		params,
		pos: jnp.ndarray,
		atoms: jnp.ndarray,
		nspins: Tuple[int,...],
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


	orbitals,(ae,r_ae,r_ee,he,hz)=fermi_net_orbitals(
		params,
		pos,
		atoms=atoms,
		nspins=nspins,
		options=options,
	)
	# print("in fermi_net,pos.shape:",pos.shape)  #(2,3)
	# print("in fermi_net,atmos.shape:",atoms.shape)  #(2,3)
	if options.envelope_pw is not None:  #options.envelope_pw:None
		# geminal planewave envelope
		phase_e,logabspsi_e=options.envelope_pw.apply(params['envelope_pw'],pos,he,hz,twist,)
		w=phase_e*jnp.exp(logabspsi_e)
		if params['det'] is not None:
			raise RuntimeError("the trainable parameters of det is not supported with pw envelope")
	else:
		if params['det'] is not None:
			w=params['det']
		else:
			w=None
	# print("w:",w)  #None
	# print("options.envelope.apply_type:",options.envelope.apply_type)
	if options.det_mode=='det':
		output=network_blocks.logdet_matmul(orbitals,w=w,do_complex=options.do_complex)
		if options.envelope.apply_type==envelopes.EnvelopeType.POST_DETERMINANT:
			output=output[0],output[1]+options.envelope.apply(ae=ae,r_ae=r_ae,r_ee=r_ee,**params['envelope'])

	elif options.det_mode=='gemi':
		#print('orbitals:',orbitals[0][0][0])
		output=options.gemi_ia[1](params['gemi'],orbitals,do_complex=options.do_complex)
		#print('output:',output)
	else:
		raise RuntimeError(f'unknown det_mode {options.det_mode}')
	return output

def fermi_orbitals(params,
		pos: jnp.ndarray,
		atoms: jnp.ndarray,
		nspins: Tuple[int,...],
		options: FermiNetOptions = FermiNetOptions(),
):
	orbitals,_=fermi_net_orbitals(
		params,
		pos,
		atoms=atoms,
		nspins=nspins,
		options=options,
	)

	return orbitals

def make_fermi_net(
		natom: int,
		ndim: int,
		nspins: Tuple[int,int],
		charges: jnp.ndarray,
		*,
		envelope: Optional[envelopes.Envelope] = None,
		feature_layer: Optional[FeatureLayer] = None,
		ferminet_model: Optional[FerminetModel] = None,
		bias_orbitals: bool = False,
		use_last_layer: bool = False,
		hf_solution=None,
		full_det: bool = True,
		hidden_dims: FermiLayers = ((256,32),(256,32),(256,32)),
		determinants: int = 16,    #1
		after_determinants: Union[int,Tuple[int,...]] = 1,

		det_nlayer: Optional[int] = None,
		do_complex: bool = False,
		numb_k: int = 0,   #1
		orb_numb_k: int = 0,
		env_twist_hiddens: Tuple[int] = (),
		do_twist: bool = False,
		do_aa: bool = False,
		mes: dp.ManyElectronSystem = None,
		det_mode: str = 'det',
		gemi_params: str = None,
		equal_footing: bool = False,
) -> Tuple[InitFermiNet,FermiNetLike,FermiNetOptions,FermiNetLike]:
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
	del after_determinants

	if not envelope:
		envelope=envelopes.make_isotropic_envelope()

	if det_mode=="gemi":
		gemi_ia=gemi.make_gemi(
			nspins,
			**gemi_params
		)
		if full_det:
			logging.info(
				"you are setting full_det in the gemi determinant mode, "
				"which does not make sense. "
				"We have changed full_det to False "
			)
			full_det=False
	else:
		gemi_ia=None


	envelope_pw=None

	orb_env_pw=None

	options=FermiNetOptions(
		hidden_dims=hidden_dims,
		use_last_layer=use_last_layer,
		determinants=determinants,    #K:1
		full_det=full_det,
		bias_orbitals=bias_orbitals,
		envelope=envelope,
		feature_layer=feature_layer,
		ferminet_model=ferminet_model,
		det_nlayer=det_nlayer,
		do_complex=do_complex,
		envelope_pw=envelope_pw,
		orb_env_pw=orb_env_pw,
		do_aa=do_aa,
		mes=mes,
		equal_footing=equal_footing,
		det_mode=det_mode,
		gemi_params=gemi_params,
		gemi_ia=gemi_ia,
	)
	'''
	print('hidden_dims=',hidden_dims,'\n',
		'use_last_layer=',use_last_layer,'\n',
		'determinants=',determinants, '\n',   #K:1
		'full_det=',full_det,'\n',
		'bias_orbitals=',bias_orbitals,'\n',
		'envelope=',envelope,'\n',
		'feature_layer=',feature_layer,'\n',
		'ferminet_model=',ferminet_model,'\n',
		'det_nlayer=',det_nlayer,'\n',
		'do_complex=',do_complex,'\n',
		'envelope_pw=',envelope_pw,'\n',
		'orb_env_pw=',orb_env_pw,'\n',
		'do_aa=',do_aa,'\n',
		'mes=',mes,'\n',
		'equal_footing=',equal_footing,'\n',
		'det_mode=',det_mode,'\n',
		'gemi_params=',gemi_params,'\n',
		'gemi_ia=',gemi_ia,)
	'''
	#hidden_dims=((10,10),(10,10),(10,10),(10,10))
	#use_last_layer=False
	#determinants=1
	#full_det=False
	#bias_orbitals=False
	#envelope=Envelope(apply_type=<EnvelopeType.PRE_DETERMINANT_Z: 3>,init=<functionmake_ds_hz_envelope.<locals>.initat0x17c0ed440>,apply=<functionmake_ds_hz_envelope.<locals>.applyat0x17c0ed620>)
	#feature_layer=FeatureLayer(init=<functionmake_open_features.<locals>.initat0x17c0dbce0>,apply=<functionmake_open_features.<locals>.apply_at0x17c0ecb80>)
	#ferminet_model=FerminetModel(init=<functionmake_fermi_net_model_zinv_shrd.<locals>.initat0x17c0ed6c0>,apply=<functionmake_fermi_net_model_zinv_shrd.<locals>.applyat0x17c0edb20>)
	#det_nlayer=None
	#do_complex=True
	#envelope_pw=None
	#orb_env_pw=None
	#do_aa=True
	#mes=<dp.ManyElectronSystemobjectat0x17c05bd90>
	#equal_footing=False
	#det_mode=gemi
	#gemi_params=diag_shift: 0.1

	init=functools.partial(
		init_fermi_net_params,
		natom=natom,
		ndim=ndim,
		nspins=nspins,
		options=options,
		hf_solution=hf_solution,
	)
	network=functools.partial(
		fermi_net,
		nspins=nspins,
		options=options,
	)
	orbitals=functools.partial(
		fermi_orbitals,
		nspins=nspins,
		options=options,
	)
	#使用了functools.partial来创建fermi_net函数的部分应用版本。这意味着，你可以预先指定一些参数（在这个例子中是nspins和options），
	#然后创建一个新的函数network，它已经包含了这些预设的参数值。当你之后调用network函数时，就只需要提供fermi_net的其余参数，
	#而不是每次都重复指定nspins和options

	return init,network,options,orbitals


