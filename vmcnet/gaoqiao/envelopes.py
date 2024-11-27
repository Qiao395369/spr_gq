# Author:Qiaoqiao
# Date:2024/3/23
# filename:envelopes
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

"""Multiplicative envelope functions."""

import enum
from typing import Any,Mapping,Optional,Sequence,Tuple,Union

import attr
import vmcnet.gaoqiao.network_blocks as network_blocks
#import sto
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
#import ds
import itertools

_MAX_POLY_ORDER=5  # highest polynomial used in envelopes


class EnvelopeType(enum.Enum):
	"""The point at which the envelope is applied."""
	PRE_ORBITAL=enum.auto()
	PRE_DETERMINANT=enum.auto()
	PRE_DETERMINANT_Z=enum.auto()
	POST_DETERMINANT=enum.auto()
	PRE_DETERMINANT_TWIST=enum.auto()


class EnvelopeLabel(enum.Enum):
	"""Available multiplicative envelope functions."""
	ISOTROPIC=enum.auto()
	DIAGONAL=enum.auto()
	FULL=enum.auto()
	NULL=enum.auto()
	STO=enum.auto()
	STO_POLY=enum.auto()
	OUTPUT=enum.auto()
	EXACT_CUSP=enum.auto()


class EnvelopeInit(Protocol):

	def __call__(
			self,
			natom: int,
			output_dims: Union[int,Sequence[int]],
			hf,
			ndim: int) -> Union[Mapping[str,Any],Sequence[Mapping[str,Any]]]:
		"""Returns the envelope parameters.

    Envelopes applied separately to each spin channel must create a sequence of
    parameters, one for each spin channel. Other envelope types must create a
    single mapping.

    Args:
      natom: Number of atoms in the system.
      output_dims: The dimension of the layer to which the envelope is applied,
        per-spin channel for pre_determinant envelopes and a scalar otherwise.
      hf: If present, initialise the parameters to match the Hartree-Fock
        solution. Otherwise a random initialisation is use. Not supported by all
        envelope types.
      ndim: Dimension of system. Change with care.
    """


class EnvelopeApply(Protocol):

	def __call__(self,*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	             **kwargs: jnp.ndarray) -> jnp.ndarray:
		"""Returns a multiplicative envelope to ensure boundary conditions are met.

    If the envelope is applied before orbital shaping or after determinant
    evaluation, the envelope function is called once and N is the number of
    electrons. If the envelope is applied after orbital shaping and before
    determinant evaluation, the envelope function is called once per spin
    channel and N is the number of electrons in the spin channel.

    The envelope applied post-determinant evaluation is assumed to be in
    log-space.

    Args:
      ae: atom-electron vectors, shape (N, natom, ndim).
      r_ae: atom-electron distances, shape (N, natom, 1).
      r_ee: electron-electron distances, shape (N, nel, 1).
      **kwargs: learnable parameters of the envelope function.
    """


@attr.s(auto_attribs=True)
class Envelope:
	apply_type: EnvelopeType
	init: EnvelopeInit
	apply: EnvelopeApply


@attr.s(auto_attribs=True)
class GemiEnvelope:
	init: EnvelopeInit
	apply: EnvelopeApply


def _apply_covariance(x: jnp.ndarray,y: jnp.ndarray) -> jnp.ndarray:
	"""Equivalent to jnp.einsum('ijk,kmjn->ijmn', x, y)."""
	# We can avoid first reshape - just make params['sigma'] rank 3
	i,_,_=x.shape
	k,m,j,n=y.shape
	x=x.transpose((1,0,2))
	y=y.transpose((2,0,1,3)).reshape((j,k,m*n))
	vdot=jax.vmap(jnp.dot,(0,0))
	return vdot(x,y).reshape((j,i,m,n)).transpose((1,0,2,3))


def make_isotropic_envelope() -> Envelope:
	"""Creates an isotropic exponentially decaying multiplicative envelope."""

	def init(natom: int,output_dims: Sequence[int],hf=None,
	         ndim: int = 3) -> Sequence[Mapping[str,jnp.ndarray]]:
		del hf,ndim  # unused
		params=[]
		for output_dim in output_dims:
			params.append({
				'pi':jnp.ones(shape=(natom,output_dim)),
				'sigma':jnp.ones(shape=(natom,output_dim))
			})
		return params

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          pi: jnp.ndarray,sigma: jnp.ndarray) -> jnp.ndarray:
		"""Computes an isotropic exponentially-decaying multiplicative envelope."""
		del ae,r_ee  # unused
		return jnp.sum(jnp.exp(-r_ae*sigma)*pi,axis=1)

	return Envelope(EnvelopeType.PRE_DETERMINANT,init,apply)


def make_diagonal_envelope() -> Envelope:
	"""Creates a diagonal exponentially-decaying multiplicative envelope."""

	def init(natom: int,output_dims: Sequence[int],hf=None,
	         ndim: int = 3) -> Sequence[Mapping[str,jnp.ndarray]]:
		del hf  # unused
		params=[]
		for output_dim in output_dims:
			params.append({
				'pi':jnp.ones(shape=(natom,output_dim)),
				'sigma':jnp.ones(shape=(natom,ndim,output_dim))
			})
		return params

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          pi: jnp.ndarray,sigma: jnp.ndarray) -> jnp.ndarray:
		"""Computes a diagonal exponentially-decaying multiplicative envelope."""
		del r_ae,r_ee  # unused
		r_ae_sigma=jnp.linalg.norm(ae[...,None]*sigma,axis=2)
		return jnp.sum(jnp.exp(-r_ae_sigma)*pi,axis=1)

	return Envelope(EnvelopeType.PRE_DETERMINANT,init,apply)


def make_full_envelope() -> Envelope:
	"""Computes a fully anisotropic exponentially-decaying envelope."""

	def init(natom: int,output_dims: Sequence[int],hf=None,
	         ndim: int = 3) -> Sequence[Mapping[str,jnp.ndarray]]:
		del hf  # unused
		eye=jnp.eye(ndim)
		params=[]
		for output_dim in output_dims:
			params.append({
				'pi':jnp.ones(shape=(natom,output_dim)),
				'sigma':jnp.tile(eye[...,None,None],[1,1,natom,output_dim])
			})
		return params

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          pi: jnp.ndarray,sigma: jnp.ndarray) -> jnp.ndarray:
		"""Computes a fully anisotropic exponentially-decaying envelope."""
		del r_ae,r_ee  # unused
		ae_sigma=_apply_covariance(ae,sigma)
		r_ae_sigma=jnp.linalg.norm(ae_sigma,axis=2)
		return jnp.sum(jnp.exp(-r_ae_sigma)*pi,axis=1)

	return Envelope(EnvelopeType.PRE_DETERMINANT,init,apply)


def make_null_envelope() -> Envelope:
	"""Creates an no-op (identity) envelope."""

	def init(natom: int,output_dims: Sequence[int],hf=None,
	         ndim: int = 3) -> Sequence[Mapping[str,jnp.ndarray]]:
		del natom,ndim,hf  # unused
		return [{} for _ in output_dims]

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,
	          r_ee: jnp.ndarray) -> jnp.ndarray:
		del ae,r_ae,r_ee
		return jnp.ones(shape=(1,))

	return Envelope(EnvelopeType.PRE_DETERMINANT,init,apply)


def make_sto_envelope() -> Envelope:
	"""Creates a Slater-type orbital envelope: exp(-sigma*r_ae) * r_ae^n * pi."""

	def init(natom: int,output_dims: int,hf=None,
	         ndim: int = 3) -> Mapping[str,jnp.ndarray]:

		pi=jnp.zeros(shape=(natom,output_dims))
		sigma=jnp.tile(jnp.eye(ndim)[...,None,None],[1,1,natom,output_dims])
		# log order of the polynomial (initialize so the order is near zero)
		n=-50*jnp.ones(shape=(natom,output_dims))

		if hf is not None:
			j=0
			for i,atom in enumerate(hf.molecule):
				coeffs=sto.STO_6G_COEFFS[atom.symbol]
				for orb in coeffs.keys():
					order=int(orb[0])-(1 if orb[1]=='s' else 2)
					log_order=jnp.log(order+jnp.exp(-50.0))
					zeta,c=coeffs[orb]
					for _ in range(1 if orb[1]=='s' else 3):
						pi=pi.at[i,j].set(c)
						n=n.at[i,j].set(log_order)
						sigma=sigma.at[...,i,j].mul(zeta)
						j+=1
		return {'pi':pi,'sigma':sigma,'n':n}

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          pi: jnp.ndarray,sigma: jnp.ndarray,n: jnp.ndarray) -> jnp.ndarray:
		"""Computes a Slater-type orbital envelope: exp(-sigma*r_ae) * r_ae^n * pi."""
		del r_ae,r_ee  # unused
		ae_sigma=_apply_covariance(ae,sigma)
		r_ae_sigma=jnp.linalg.norm(ae_sigma,axis=2)
		exp_r_ae=jnp.exp(-r_ae_sigma+jnp.exp(n)*jnp.log(r_ae_sigma))
		out=jnp.sum(exp_r_ae*pi,axis=1)
		return out

	return Envelope(EnvelopeType.PRE_ORBITAL,init,apply)


def make_sto_poly_envelope() -> Envelope:
	"""Creates a Slater-type orbital envelope."""

	def init(natom: int,output_dims: int,hf=None,
	         ndim: int = 3) -> Mapping[str,jnp.ndarray]:

		pi=jnp.zeros(shape=(natom,output_dims,_MAX_POLY_ORDER))
		sigma=jnp.tile(jnp.eye(ndim)[...,None,None],[1,1,natom,output_dims])

		if hf is not None:
			# Initialize envelope to match basis set elements.
			j=0
			for i,atom in enumerate(hf.molecule):
				coeffs=sto.STO_6G_COEFFS[atom.symbol]
				for orb in coeffs.keys():
					order=int(orb[0])-(1 if orb[1]=='s' else 2)
					zeta,c=coeffs[orb]
					for _ in range(1 if orb[1]=='s' else 3):
						pi=pi.at[i,j,order].set(c)
						sigma=sigma.at[...,i,j].mul(zeta)
						j+=1
		return {'pi':pi,'sigma':sigma}

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          pi: jnp.ndarray,sigma: jnp.ndarray) -> jnp.ndarray:
		"""Computes a Slater-type orbital envelope."""
		del r_ae,r_ee  # unused
		# Should register KFAC tags and blocks.
		# Envelope: exp(-sigma*r_ae) * (sum_i r_ae^i * pi_i)
		ae_sigma=_apply_covariance(ae,sigma)
		r_ae_sigma=jnp.linalg.norm(ae_sigma,axis=2)
		exp_r_ae=jnp.exp(-r_ae_sigma)
		poly_r_ae=jnp.power(
			jnp.expand_dims(r_ae_sigma,-1),jnp.arange(_MAX_POLY_ORDER))
		out=jnp.sum(exp_r_ae*jnp.sum(poly_r_ae*pi,axis=3),axis=1)
		return out

	return Envelope(EnvelopeType.PRE_ORBITAL,init,apply)


def make_output_envelope() -> Envelope:
	"""Creates an anisotropic multiplicative envelope to apply to determinants."""

	def init(natom: int,output_dims: int,hf=None,
	         ndim: int = 3) -> Mapping[str,jnp.ndarray]:
		"""Initialise learnable parameters for output envelope."""
		del output_dims,hf  # unused
		return {
			'pi':jnp.zeros(shape=natom),
			'sigma':jnp.tile(jnp.eye(ndim)[...,None],[1,1,natom])
		}

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          pi: jnp.ndarray,sigma: jnp.ndarray) -> jnp.ndarray:
		"""Fully anisotropic envelope, but only one output in log space."""
		del r_ae,r_ee  # unused
		# Should register KFAC tags and blocks.
		sigma=jnp.expand_dims(sigma,-1)
		ae_sigma=jnp.squeeze(_apply_covariance(ae,sigma),axis=-1)
		r_ae_sigma=jnp.linalg.norm(ae_sigma,axis=2)
		return jnp.sum(jnp.log(jnp.sum(jnp.exp(-r_ae_sigma+pi),axis=1)))

	return Envelope(EnvelopeType.POST_DETERMINANT,init,apply)


def make_exact_cusp_envelope(nspins: Tuple[int,int],
                             charges: jnp.ndarray) -> Envelope:
	"""Creates an envelope satisfying cusp conditions to apply to determinants."""

	def init(natom: int,output_dims: int,hf=None,
	         ndim: int = 3) -> Mapping[str,jnp.ndarray]:
		"""Initialise learnable parameters for the exact cusp envelope."""
		del output_dims,hf  # unused
		return {
			'pi':jnp.zeros(shape=natom),
			'sigma':jnp.tile(jnp.eye(ndim)[...,None],[1,1,natom])
		}

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          pi: jnp.ndarray,sigma: jnp.ndarray) -> jnp.ndarray:
		"""Combine exact cusp conditions and envelope on the output into one."""
		# No cusp at zero
		del r_ae  # unused
		# Should register KFAC tags and blocks.
		sigma=jnp.expand_dims(sigma,-1)
		ae_sigma=jnp.squeeze(_apply_covariance(ae,sigma),axis=-1)
		soft_r_ae=jnp.sqrt(jnp.sum(1.+ae_sigma**2,axis=2))
		env=jnp.sum(jnp.log(jnp.sum(jnp.exp(-soft_r_ae+pi),axis=1)))

		# atomic cusp
		r_ae=jnp.linalg.norm(ae,axis=2)
		a_cusp=jnp.sum(charges/(1.+r_ae))

		# electronic cusp
		spin_partitions=network_blocks.array_partitions(nspins)
		r_ees=[
			jnp.split(r,spin_partitions,axis=1)
			for r in jnp.split(r_ee,spin_partitions,axis=0)
		]
		# Sum over same-spin electrons twice but different-spin once, which
		# cancels out the different factor of 1/2 and 1/4 in the cusps.
		e_cusp=(
				jnp.sum(1./(1.+r_ees[0][0]))+jnp.sum(1./(1.+r_ees[1][1]))+
				jnp.sum(1./(1.+r_ees[0][1])))
		return env+a_cusp-0.5*e_cusp

	return Envelope(EnvelopeType.POST_DETERMINANT,init,apply)


def get_envelope(
		envelope_label: EnvelopeLabel,
		**kwargs: Any,
) -> Envelope:
	"""Gets the desired multiplicative envelope function.

  Args:
    envelope_label: envelope function required.
    **kwargs: keyword arguments forwarded to the envelope.

  Returns:
    (envelope_type, envelope), where envelope_type describes when the envelope
    should be applied in the network and envelope is the envelope function.
  """
	envelope_builders={
		EnvelopeLabel.STO:make_sto_envelope,
		EnvelopeLabel.STO_POLY:make_sto_poly_envelope,
		EnvelopeLabel.ISOTROPIC:make_isotropic_envelope,
		EnvelopeLabel.DIAGONAL:make_diagonal_envelope,
		EnvelopeLabel.FULL:make_full_envelope,
		EnvelopeLabel.NULL:make_null_envelope,
		EnvelopeLabel.OUTPUT:make_output_envelope,
		EnvelopeLabel.EXACT_CUSP:make_exact_cusp_envelope,
	}
	return envelope_builders[envelope_label](**kwargs)


def make_multiwave_envelope(kpoints: jnp.ndarray) -> Envelope:
	"""Returns an oscillatory envelope.

  Envelope consists of a sum of truncated 3D Fourier series, one centered on
  each atom, with Fourier frequencies given by kpoints:

    sigma_{2i}*cos(kpoints_i.r_{ae}) + sigma_{2i+1}*sin(kpoints_i.r_{ae})

  Initialization sets the coefficient of the first term in each
  series to 1, and all other coefficients to 0. This corresponds to the
  cosine of the first entry in kpoints. If this is [0, 0, 0], the envelope
  will evaluate to unity at the beginning of training.

  Args:
    kpoints: Reciprocal lattice vectors of terms included in the Fourier
      series. Shape (nkpoints, ndim) (Note that ndim=3 is currently
      a hard-coded default).

  Returns:
    An instance of ferminet.envelopes.Envelope with apply_type
    envelopes.EnvelopeType.PRE_DETERMINANT
  """

	def init(natom: int,
	         output_dims: Sequence[int],
	         hf=None,
	         ndim: int = 3) -> Sequence[Mapping[str,jnp.ndarray]]:
		"""See ferminet.envelopes.EnvelopeInit."""
		del hf,natom,ndim  # unused
		params=[]
		nk=kpoints.shape[0]
		for output_dim in output_dims:
			params.append({'sigma':jnp.zeros((2*nk,output_dim))})
			params[-1]['sigma']=params[-1]['sigma'].at[0,:].set(1.0)
		return params

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          sigma: jnp.ndarray) -> jnp.ndarray:
		"""See ferminet.envelopes.EnvelopeApply."""
		del r_ae,r_ee  # unused
		phase_coords=ae@kpoints.T
		waves=jnp.concatenate((jnp.cos(phase_coords),jnp.sin(phase_coords)),
		                      axis=2)
		env=waves@sigma
		return jnp.sum(env,axis=1)

	return Envelope(EnvelopeType.PRE_DETERMINANT,init,apply)


def make_kpoints(lattice: jnp.ndarray,
                 spins: Tuple[int,int],
                 min_kpoints: Optional[int] = None) -> jnp.ndarray:
	"""Generates an array of reciprocal lattice vectors.

  Args:
    lattice: Matrix whose columns are the primitive lattice vectors of the
      system, shape (ndim, ndim). (Note that ndim=3 is currently
      a hard-coded default).
    spins: Tuple of the number of spin-up and spin-down electrons.
    min_kpoints: If specified, the number of kpoints which must be included in
      the output. The number of kpoints returned will be the
      first filled shell which is larger than this value. Defaults to None,
      which results in min_kpoints == sum(spins).

  Raises:
    ValueError: Fewer kpoints requested by min_kpoints than number of
      electrons in the system.

  Returns:
    jnp.ndarray, shape (nkpoints, ndim), an array of reciprocal lattice
      vectors sorted in ascending order according to length.
  """
	rec_lattice=2*jnp.pi*jnp.linalg.inv(lattice)
	# Calculate required no. of k points
	if min_kpoints is None:
		min_kpoints=sum(spins)
	elif min_kpoints<sum(spins):
		raise ValueError(
			'Number of kpoints must be equal or greater than number of electrons')

	dk=1+1e-5
	# Generate ordinals of the lowest min_kpoints kpoints
	max_k=int(jnp.ceil(min_kpoints*dk)**(1/3.))
	ordinals=sorted(range(-max_k,max_k+1),key=abs)
	ordinals=jnp.asarray(list(itertools.product(ordinals,repeat=3)))

	kpoints=ordinals@rec_lattice.T
	kpoints=jnp.asarray(sorted(kpoints,key=jnp.linalg.norm))
	k_norms=jnp.linalg.norm(kpoints,axis=1)

	return kpoints[k_norms<=k_norms[min_kpoints-1]*dk]


def make_pbc_full_nn(
		lattice: jnp.ndarray,
		rc: float,
		rc_smth: float,
) -> Envelope:
	"""Returns a full envelope for pbc systems. only search for the nearest neighbors
  """
	batch_apply_covar=jax.vmap(_apply_covariance,in_axes=[0,None])
	batch_multiply=jax.vmap(jnp.multiply,in_axes=[0,None])
	rec_lattice=jnp.linalg.inv(lattice)
	if not dp.auto_nearest_neighbor(lattice,rc):
		raise RuntimeError('the rc should be no larger than half box length')

	def init(natom: int,output_dims: Sequence[int],hf=None,
	         ndim: int = 3) -> Sequence[Mapping[str,jnp.ndarray]]:
		del hf  # unused
		eye=jnp.eye(ndim)
		params=[]
		for output_dim in output_dims:
			params.append({
				'pi':jnp.ones(shape=(natom,output_dim)),
				'sigma':jnp.tile(eye[...,None,None],[1,1,natom,output_dim])
			})
		return params

	def decayed_exp(xx,rr):
		sw=dp.switch_func_poly(rr,rc,rc_smth)
		exp=jnp.exp(-xx)
		return sw*exp

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          pi: jnp.ndarray,sigma: jnp.ndarray) -> jnp.ndarray:
		"""Computes a fully anisotropic exponentially-decaying envelope."""
		del r_ee  # unused
		ae=dp.apply_nearest_neighbor(ae,lattice,rec_lattice)
		ae_sigma=envelopes._apply_covariance(ae,sigma)
		# nele x nion x 3 x orb_dim
		ae_sigma=ferminet.curvature_tags_and_blocks.register_qmc(
			ae_sigma,ae,sigma,type='full')
		# nele x nion x orb_dim
		r_ae_sigma=jnp.linalg.norm(ae_sigma,axis=2)
		# # nele x orb_dim
		return jnp.sum(decayed_exp(r_ae_sigma,r_ae)*pi,axis=1)

	return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT,init,apply)


def make_ds_isotropic_envelope(
		lattice: jnp.ndarray,
) -> Envelope:
	"""Creates an isotropic exponentially decaying multiplicative envelope."""
	org_lattice=lattice/(2.*jnp.pi)
	rec_lattice=jnp.linalg.inv(org_lattice)

	def init(natom: int,output_dims: Sequence[int],hf=None,
	         ndim: int = 3) -> Sequence[Mapping[str,jnp.ndarray]]:
		del hf,ndim  # unused
		params=[]
		for output_dim in output_dims:
			params.append({
				'pi':jnp.ones(shape=(natom,output_dim)),
				'sigma':jnp.ones(shape=(natom,output_dim))
			})
		return params

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          pi: jnp.ndarray,sigma: jnp.ndarray) -> jnp.ndarray:
		"""Computes an isotropic exponentially-decaying multiplicative envelope."""
		del r_ae,r_ee  # unused
		# ne x na
		prim_periodic_sea,_=ds.nu_distance(ae,org_lattice,rec_lattice)
		# ne x na x None multiply na x odim -> ne x na x odim
		# sum(axis=1) -> ne x odim
		return jnp.sum(jnp.exp(-prim_periodic_sea[:,:,None]*sigma)*pi,axis=1)

	return Envelope(EnvelopeType.PRE_DETERMINANT,init,apply)


def make_ds_hz_envelope(
		hiddens: Tuple[int] = (8,8),  #[8]
) -> Envelope:
	"""Creates an isotropic exponentially decaying multiplicative envelope."""

	def init_one(
			key,
			hz_size,  #16
			output_dim, #odim+diff or odim
	) -> Sequence[Mapping[str,jnp.ndarray]]:
		dims_in=[hz_size]+hiddens  #[16,8]
		#print('heddens:',hiddens) #hiddens=8
		## last dim: pi and sigma
		dims_out=hiddens+[output_dim] #[8,16]
		params={}
		params['pi']=[]
		params['sigma']=[]
		for ii in range(len(dims_in)):
			key,subkey=jax.random.split(key)
			params['pi'].append(
				network_blocks.init_linear_layer(
					subkey,
					in_dim=dims_in[ii],
					out_dim=dims_out[ii],
					include_bias=True,
					scale=0.0,
				))
			key,subkey=jax.random.split(key)
			params['sigma'].append(
				network_blocks.init_linear_layer(
					subkey,
					in_dim=dims_in[ii],
					out_dim=dims_out[ii],
					include_bias=True,
					scale=0.0,
				))
		return params

	def init(
			key,
			hz_size,  #(16)
			output_dims, #[16,16] or [odim+diff,odim]
	):
		params=[]
		for output_dim in output_dims:
			params.append(
				init_one(key,hz_size,output_dim)
			)
		return params

	residual=lambda x,y:(x+y)/jnp.sqrt(2.0) if x.shape==y.shape else y

	def apply_net(
			hz,params,
	):
		hz_in=hz
		for ii in range(len(params)):
			hz_out=network_blocks.linear_layer(hz_in,**(params[ii]))
			if ii!=len(params)-1:
				hz_out=jnp.tanh(hz_out)
			hz_in=residual(hz_in,hz_out)
		return hz_in

	def apply(
			*,
			hz,  #hz:(14,16)质子单流
			ae,  #(7,14,3)输入的是某种自旋的7个电子和14个质子的位移差
			pi,
			sigma,
	) -> jnp.ndarray:
		"""Computes an isotropic exponentially-decaying multiplicative envelope."""
		# hz: natoms x nfz
		# natoms x dims_out
		pi=apply_net(hz,pi)   #得到的就是pi:(14,16)
		sigma=apply_net(hz,sigma)       #得到的是sigma:(14,16)
		pi=1.0+1.*pi
		sigma=0.5+1.*sigma
		#print('sigma.shape:',sigma.shape)
		# ne x na
		#prim_periodic_sea,_=ds.nu_distance(ae,org_lattice,rec_lattice)  #(7,14)
		r_ae=jnp.linalg.norm(ae,axis=-1)
		# ne x na x None multiply na x odim -> ne x na x odim
		# sum(axis=1) -> ne x odim
		aa=jnp.sum(jnp.exp(-r_ae[:,:,None]*sigma)*pi,axis=1)
		#print('aa.shape:',aa.shape)
		return aa
		#(7,14,1)*(14,16)->(7,14,16)-=>(7,16)

	return Envelope(EnvelopeType.PRE_DETERMINANT_Z,init,apply)


def ds_allsame_isotropic_envelope(
		lattice: jnp.ndarray,
) -> Envelope:
	"""Creates an isotropic exponentially decaying multiplicative envelope."""
	org_lattice=lattice/(2.*jnp.pi)
	rec_lattice=jnp.linalg.inv(org_lattice)

	def init(natom: int,output_dims: Sequence[int],hf=None,
	         ndim: int = 3) -> Sequence[Mapping[str,jnp.ndarray]]:
		del hf,ndim  # unused
		params=[]
		for output_dim in output_dims:
			params.append({
				'pi':jnp.ones(shape=(output_dim)),
				'sigma':jnp.ones(shape=(output_dim))
			})
		return params

	def apply(*,ae: jnp.ndarray,r_ae: jnp.ndarray,r_ee: jnp.ndarray,
	          pi: jnp.ndarray,sigma: jnp.ndarray) -> jnp.ndarray:
		"""Computes an isotropic exponentially-decaying multiplicative envelope."""
		del r_ae,r_ee  # unused
		# ne x na
		prim_periodic_sea,_=ds.nu_distance(ae,org_lattice,rec_lattice)
		# ne x na x None multiply na x odim -> ne x na x odim
		# sum(axis=1) -> ne x odim
		return jnp.sum(jnp.exp(-prim_periodic_sea[:,:,None]*sigma[None,None,:])*pi[None,None,:],axis=1)

	return Envelope(EnvelopeType.PRE_DETERMINANT,init,apply)


def make_planewave_envelope_with_ion(
		nspins: Tuple[int,int],
		lattice: jnp.ndarray,
		numb_k: int = 0,
) -> Envelope:
	nele=sum(nspins)
	numb_k=nele if numb_k<nele else numb_k
	kpoints=make_kpoints(lattice,nspins,numb_k)
	nk=kpoints.shape[0]
	rec_lattice=jnp.linalg.inv(lattice)
	ndet=1
	f1=jnp.concatenate([jnp.ones([nele//2]),jnp.zeros([nk-nele//2])]).reshape([-1,1])
	invsqrtv=1./(jnp.linalg.det(lattice)**0.5)

	def init(
			key,
			flo: float = -0.2,
			fhi: float = 0.2,
	) -> Sequence[Mapping[str,jnp.ndarray]]:
		params=f1+jax.random.uniform(key,shape=(nk,1))*(fhi-flo)+flo
		return params

	def apply(
			params,
			ae,
			twist,
	) -> jnp.ndarray:
		kpt=kpoints+2.*jnp.pi*jnp.matmul(twist,rec_lattice.T)[None,:]
		# ne x a x 3
		(ne,na,_)=ae.shape
		# ne x na x nk
		zkiI=invsqrtv*jnp.exp(1j*jnp.dot(ae,kpt.T))
		# ne x nk
		zki=jnp.mean(zkiI,axis=1)
		# split trick adjust to kfac
		Ds=jnp.matmul(jnp.concatenate([zki.real,zki.imag],axis=0),params)
		Ds=jnp.split(Ds,[ne],axis=0)
		# ne x 1
		D=Ds[0]+1j*Ds[1]
		ret=jnp.sum(jnp.log(D))
		phase,logabs=jnp.exp(1j*ret.imag),ret.real
		return phase,logabs

	return GemiEnvelope(init,apply)


def make_planewave_envelope(
		nspins: Tuple[int,int],
		lattice: jnp.ndarray,
		numb_k: int = 0,
		ndet: int = 1,
		hiddens: Tuple[int] = None,
) -> Envelope:
	nele=sum(nspins)
	kpoints=make_kpoints(lattice,nspins,nele if numb_k<nele else numb_k)
	if numb_k<nele:
		kpoints=kpoints[:numb_k]
	nk=kpoints.shape[0]
	rec_lattice=jnp.linalg.inv(lattice)
	if nk>=nele:
		f1=jnp.concatenate([jnp.ones([nele//2,ndet]),
		                    jnp.zeros([nk-nele//2,ndet])]).reshape([nk,ndet])
	else:
		f1=jnp.ones([nk,ndet])
	invsqrtv=1./(jnp.linalg.det(lattice)**0.5)
	residual=lambda x,y:(x+y)/jnp.sqrt(2.0) if x.shape==y.shape else y
	do_twist_mlp=(hiddens is not None)

	def init(
			key,
			dims_orbital_in,
			dims_z,
			flo: float = -0.1,
			fhi: float = 0.1,
			jastrow_init_scale: float = 0.01,
	) -> Sequence[Mapping[str,jnp.ndarray]]:
		params={}
		key,subkey=jax.random.split(key,num=2)
		params['proj_h']=network_blocks.init_linear_layer(subkey,dims_orbital_in,3)
		key,subkey=jax.random.split(key,num=2)
		params['z']=network_blocks.init_linear_layer(subkey,dims_z,ndet)
		params['z']['w']*=jastrow_init_scale
		params['z']['b']*=jastrow_init_scale
		if do_twist_mlp:
			in_dims=[3]+list(hiddens)
			out_dims=list(hiddens)+[nk*ndet]
			params['f']=[]
			for jj in range(len(in_dims)):
				key,subkey=jax.random.split(key,num=2)
				params['f'].append(
					network_blocks.init_linear_layer(
						subkey,
						in_dim=in_dims[jj],
						out_dim=out_dims[jj],
						include_bias=True))
		else:
			key,subkey=jax.random.split(key,num=2)
			params['f']=f1+jax.random.uniform(subkey,shape=(nk,ndet))*(fhi-flo)+flo
		return params

	def apply(
			params,
			pos,
			he,
			hz,
			twist,
	) -> jnp.ndarray:
		# nz x ndet
		hz=network_blocks.linear_layer(hz,**params['z'])
		# ndet
		jastrow=jnp.sum(hz,axis=0)

		# project h1 to R3
		hx=network_blocks.linear_layer(he,**params['proj_h'])
		# backflow
		xe=pos.reshape([-1,3])+hx
		# planewave
		kpt=kpoints+2.*jnp.pi*jnp.matmul(twist,rec_lattice.T)[None,:]
		# ne x 3
		(ne,_)=xe.shape
		# ne x nk
		xk=jnp.dot(xe,kpt.T)
		# 1 x nk
		ekr=invsqrtv*jnp.exp(1j*jnp.sum(xk,axis=0)).reshape([1,-1])
		if do_twist_mlp:
			# twist to coefficients
			coeff=twist.reshape([1,-1])
			for ii in range(len(params['f'])):
				coeff_next=jnp.tanh(network_blocks.linear_layer(coeff,**params['f'][ii]))
				coeff=residual(coeff,coeff_next)
			coeff=coeff.reshape([nk,-1])
		else:
			coeff=params['f']
		# split trick adjust to kfac, [1 x ndet, 1 x ndet]
		Ds=jnp.matmul(jnp.concatenate([ekr.real,ekr.imag],axis=0),coeff)/jnp.sqrt(nk)
		Ds=jnp.split(Ds,[1],axis=0)
		# 1 x ndet
		D=Ds[0]+1j*Ds[1]
		# ndet
		ret=jnp.log(D.reshape((ndet)))
		phase,logabs=jnp.exp(1j*ret.imag),ret.real

		# apply jastrow e^J * Psi
		logabs=logabs+jastrow

		return phase,logabs

	return GemiEnvelope(init,apply)


def zero_out_orbital_k_network(
		prefs,
		nk,
		odim,
		include_bias=True,
):
	"""
  Keeps the first nk outputs while zero-out the rest (nk-1)*odim outputs
  """
	oldw=prefs['w']
	neww=jnp.concatenate([
		jnp.split(oldw,[odim],axis=1)[0],
		jnp.zeros((oldw.shape[0],(nk-1)*odim)),
	],axis=1)
	prefs['w']=neww
	if include_bias:
		oldb=prefs['b']
		newb=jnp.concatenate([
			jnp.split(oldb,[odim],axis=0)[0],
			jnp.zeros(((nk-1)*odim)),
		],axis=0)
		prefs['b']=newb
	return prefs


def make_planewave_orbitals(
		nspins: Tuple[int,int],
		lattice: jnp.ndarray,
		numb_k: int = 0,
		hiddens: Tuple[int] = None,
		zero_nk_lt_0: bool = True,
) -> Envelope:
	rec_lattice=jnp.linalg.inv(lattice)
	nele=sum(nspins)
	kpoints=make_kpoints(lattice,nspins,nele if numb_k<nele else numb_k)
	if numb_k<nele:
		kpoints=kpoints[:numb_k]
	nk=kpoints.shape[0]
	invsqrtv=1./(jnp.linalg.det(lattice)**0.5)
	residual=lambda x,y:(x+y)/jnp.sqrt(2.0) if x.shape==y.shape else y
	do_twist_mlp=(hiddens is not None)

	def init(
			key,
			dims_orbital_in: int,
			output_dims: Sequence[int],
			ndim: int = 3,
	) -> Sequence[Mapping[str,jnp.ndarray]]:
		params=[]
		for ii in output_dims:
			if do_twist_mlp:
				in_dims=[3]+list(hiddens)
				out_dims=list(hiddens)+[nk*ii]
				pref_list=[]
				for jj in range(len(in_dims)):
					key,subkey=jax.random.split(key,num=2)
					pref_list.append(
						network_blocks.init_linear_layer(
							subkey,
							in_dim=in_dims[jj],
							out_dim=out_dims[jj],
							include_bias=True))
					# handel the last mapping
					if jj==len(in_dims)-1 and zero_nk_lt_0:
						pref_list[-1]=zero_out_orbital_k_network(
							pref_list[-1],nk,ii,include_bias=True,
						)
			else:
				pref_list=jnp.concatenate(
					(jnp.ones([1,ii]),jnp.zeros([nk-1,ii])),axis=0,
				)
			# nk x odim
			key,subkey=jax.random.split(key,num=2)
			params.append({
				'pref':pref_list,
				'proj_h':network_blocks.init_linear_layer(
					subkey,dims_orbital_in,ndim,
				),
			})
		return params

	def apply(
			*,
			pos,
			he,
			twist,
			pref,
			proj_h,
	) -> jnp.ndarray:
		# project h1 to R3
		hx=network_blocks.linear_layer(he,**proj_h)
		# backflow
		xe=pos.reshape([-1,3])+hx
		ne=xe.shape[0]
		# planewave
		kpt=kpoints+2.*jnp.pi*jnp.matmul(twist,rec_lattice.T)[None,:]
		# ne x 3
		(ne,_)=xe.shape
		# ne x nk
		xk=jnp.dot(xe,kpt.T)
		# ne x nk
		exk=invsqrtv*jnp.exp(1j*xk)
		if do_twist_mlp:
			# twist to coefficients
			coeff=twist.reshape([1,-1])
			for ii in range(len(pref)):
				coeff_next=jnp.tanh(network_blocks.linear_layer(coeff,**pref[ii]))
				coeff=residual(coeff,coeff_next)
			coeff=coeff.reshape([nk,-1])
		else:
			coeff=pref
		# split trick adjust to kfac, [ne x odim, ne x odim]
		envs=jnp.matmul(jnp.concatenate([exk.real,exk.imag],axis=0),coeff)/jnp.sqrt(nk)
		envs=jnp.split(envs,[ne],axis=0)
		# ne x odim
		env=envs[0]+1j*envs[1]
		return env

	return Envelope(EnvelopeType.PRE_DETERMINANT,init,apply)


def make_planewave_orbitals_hz(
		nspins: Tuple[int,int],
		lattice: jnp.ndarray,
		numb_k: int = 0,
		hiddens: Tuple[int] = None,
		zero_nk_lt_0: bool = True,
) -> Envelope:
	rec_lattice=jnp.linalg.inv(lattice)
	nele=sum(nspins)
	kpoints=make_kpoints(lattice,nspins,nele if numb_k<nele else numb_k)
	if numb_k<nele:
		kpoints=kpoints[:numb_k]
	nk=kpoints.shape[0]
	invsqrtv=1./(jnp.linalg.det(lattice)**0.5)
	residual=lambda x,y:(x+y)/jnp.sqrt(2.0) if x.shape==y.shape else y
	do_twist_mlp=(hiddens is not None)

	def init(
			key,
			dims_orbital_in: int,
			output_dims: Sequence[int],
			dim_hz: int,
			ndim: int = 3,
	) -> Sequence[Mapping[str,jnp.ndarray]]:
		params=[]
		for ii in output_dims:
			if do_twist_mlp:
				in_dims=[3+dim_hz]+list(hiddens)
				out_dims=list(hiddens)+[nk*ii]
				pref_list=[]
				for jj in range(len(in_dims)):
					key,subkey=jax.random.split(key,num=2)
					pref_list.append(
						network_blocks.init_linear_layer(
							subkey,
							in_dim=in_dims[jj],
							out_dim=out_dims[jj],
							include_bias=True))
					# handel the last mapping
					if jj==len(in_dims)-1 and zero_nk_lt_0:
						pref_list[-1]=zero_out_orbital_k_network(
							pref_list[-1],nk,ii,include_bias=True,
						)
			else:
				pref_list=jnp.concatenate(
					(jnp.ones([1,ii]),jnp.zeros([nk-1,ii])),axis=0,
				)
			# nk x odim
			key,subkey=jax.random.split(key,num=2)
			params.append({
				'pref':pref_list,
				'proj_h':network_blocks.init_linear_layer(
					subkey,dims_orbital_in,ndim,
				),
			})
		return params

	def apply(
			*,
			pos,  # nele x 3
			atm,  # nz x 3
			he,  # nele x nh1
			hz,  # nz x nh1
			twist,
			pref,
			proj_h,
	) -> jnp.ndarray:
		_,ndim=pos.shape
		ae=jnp.reshape(pos,[-1,1,ndim])-atm[None,...]
		# project h1 to R3, ne x 3
		hx=network_blocks.linear_layer(he,**proj_h)
		# backflow
		xe=ae[:,:,:]+hx[:,None,:]
		ne=xe.shape[0]
		# planewave
		kpt=kpoints+2.*jnp.pi*jnp.matmul(twist,rec_lattice.T)[None,:]
		# ne x 3
		(ne,nz,_)=ae.shape
		# ne x nz x nk
		xk=jnp.einsum("izd,kd->izk",ae,kpt)
		# ne x nz x nk
		exk=invsqrtv*jnp.exp(1j*xk)

		# coeff: nk x odim
		def eval_coeff(twist,hz):
			coeff=jnp.concatenate([
				twist.reshape([1,-1]),
				hz.reshape([1,-1]),
			],axis=1)
			for ii in range(len(pref)):
				coeff_next=jnp.tanh(network_blocks.linear_layer(coeff,**pref[ii]))
				coeff=residual(coeff,coeff_next)
			# nk x odim
			coeff=coeff.reshape([nk,-1])
			return coeff

		# (nt, nfz) -> nk x odim
		# (nt, nz x nfz) -> nz x nk x odim
		vmap_eval_coeff=jax.vmap(eval_coeff,in_axes=(None,0))
		if do_twist_mlp:
			# twist to coefficients nz x nk x odim
			coeff=vmap_eval_coeff(twist,hz)
		else:
			raise RuntimeError("not supported!")
		# split trick adjust to kfac, [ne x nz x nk, ne x nz x nk] -> 2ne x nz x nk
		# 2ne x nz x nk
		exk_tmp=jnp.concatenate([exk.real,exk.imag],axis=0)
		# 2ne x nz x odim
		envs=jnp.einsum("izk,zko->izo",exk_tmp,coeff)/jnp.sqrt(nk)
		# [2ne x odim]
		envs=jnp.mean(envs,axis=1)
		# [ne x odim, ne x odim]
		envs=jnp.split(envs,[ne],axis=0)
		# ne x odim
		env=envs[0]+1j*envs[1]
		return env

	return Envelope(EnvelopeType.PRE_DETERMINANT,init,apply)


def make_copied_atom_isotropic_envelope(
		lattice: jnp.ndarray,
		ncopy: Sequence[int] = (8,8,8),
) -> Envelope:
	"""Creates an isotropic exponentially decaying multiplicative envelope."""
	rec_lattice=jnp.linalg.inv(lattice)
	ordinals=jnp.asarray(list(itertools.product(
		range(-ncopy[0],ncopy[0]+1),
		range(-ncopy[1],ncopy[1]+1),
		range(-ncopy[2],ncopy[2]+1),
	)))
	# nk x 3
	shifts=ordinals@lattice
	nk=shifts.shape[0]

	def init(
			natom: int,
			output_dims: Sequence[int],
			hf=None,
			ndim: int = 3,
	) -> Sequence[Mapping[str,jnp.ndarray]]:
		del hf,ndim  # unused
		params=[]
		for output_dim in output_dims:
			params.append({
				'pi':jnp.ones(shape=(1,output_dim)),
				'sigma':jnp.ones(shape=(1,output_dim))
			})
		return params

	def apply_one(
			ae,
			pi,
			sigma,
			twist,
	):
		# nk x 3
		sae=ae[None,:]-shifts[:,:]
		# nk
		sae=jnp.linalg.norm(sae,axis=-1).reshape([-1,1])
		# nk x 1 x odim
		expsae=jnp.exp(-sae[:,:,None]*sigma)*pi
		# nk x odim
		expsae=jnp.reshape(expsae,[nk,-1])
		# nk
		phase=jnp.dot(ordinals,twist)
		prefc=jnp.exp(2.*jnp.pi*1j*phase)
		# nk x odim
		expsae=expsae[:,:]*prefc[:,None]
		# odim
		return jnp.sum(expsae,axis=0)

	batch_apply_one=jax.vmap(apply_one,in_axes=(0,None,None,None))

	def apply(
			*,
			ae: jnp.ndarray,
			r_ae: jnp.ndarray,
			r_ee: jnp.ndarray,
			pi: jnp.ndarray,
			sigma: jnp.ndarray,
			twist: jnp.ndarray,
	) -> jnp.ndarray:
		"""Computes an isotropic exponentially-decaying multiplicative envelope."""
		del r_ae,r_ee  # unused
		ne,na,_=ae.shape
		nodim=pi.shape[-1]

		# inner function accumulate the sum over atoms
		def inner_fun(ii,res):
			res+=batch_apply_one(ae[:,ii,:],pi,sigma,twist)
			return res

		sumexp=1j*jnp.zeros([ne,nodim])
		# ne x nodim
		sumexp=jax.lax.fori_loop(0,na,inner_fun,sumexp)
		return sumexp

	return Envelope(EnvelopeType.PRE_DETERMINANT_TWIST,init,apply)
