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
# limitations under the License

"""Feature layer for periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

from typing import Optional,Tuple,Union,List
import jax.numpy as jnp

import vmcnet.gaoqiao.networks as networks
import vmcnet.gaoqiao.dp as dp
#import ds
# import decay


def make_pbc_feature_layer(charges: Optional[jnp.ndarray] = None,
                           nspins: Optional[Tuple[int,...]] = None,
                           ndim: int = 3,
                           lattice: Optional[jnp.ndarray] = None,
                           include_r_ae: bool = True) -> networks.FeatureLayer:
	"""Returns the init and apply functions for periodic features.

	Args:
		charges: (natom) array of atom nuclear charges.
		nspins: tuple of the number of spin-up and spin-down electrons.
		ndim: dimension of the system.
		lattice: Matrix whose columns are the primitive lattice vectors of the
		  system, shape (ndim, ndim).
		include_r_ae: Flag to enable electron-atom distance features. Set to False
		  to avoid cusps with ghost atoms in, e.g., homogeneous electron gas.
	"""

	del charges,nspins

	if lattice is None:
		lattice=jnp.eye(ndim)

	# Calculate reciprocal vectors, factor 2pi omitted
	reciprocal_vecs=jnp.linalg.inv(lattice)
	lattice_metric=lattice.T@lattice

	def periodic_norm(vec,metric):
		a=(1-jnp.cos(2*jnp.pi*vec))
		b=jnp.sin(2*jnp.pi*vec)
		# i,j = nelectron, natom for ae
		cos_term=jnp.einsum('ijm,mn,ijn->ij',a,metric,a)
		sin_term=jnp.einsum('ijm,mn,ijn->ij',b,metric,b)
		return (1/(2*jnp.pi))*jnp.sqrt(cos_term+sin_term)

	def init() -> Tuple[Tuple[int,int],networks.Param]:
		if include_r_ae:
			return (2*ndim+1,2*ndim+1),{}
		else:
			return (2*ndim,2*ndim+1),{}

	def apply(ae,r_ae,ee,r_ee) -> Tuple[jnp.ndarray,jnp.ndarray]:
		# One e features in phase coordinates, (s_ae)_i = k_i . ae
		s_ae=jnp.einsum('il,jkl->jki',reciprocal_vecs,ae)
		# Two e features in phase coordinates
		s_ee=jnp.einsum('il,jkl->jki',reciprocal_vecs,ee)
		# Periodized features
		ae=jnp.concatenate(
			(jnp.sin(2*jnp.pi*s_ae),jnp.cos(2*jnp.pi*s_ae)),axis=-1)
		ee=jnp.concatenate(
			(jnp.sin(2*jnp.pi*s_ee),jnp.cos(2*jnp.pi*s_ee)),axis=-1)
		# Distance features defined on orthonormal projections
		r_ae=periodic_norm(s_ae,lattice_metric)
		# Don't take gradients through |0|
		n=ee.shape[0]
		s_ee+=jnp.eye(n)[...,None]
		r_ee=periodic_norm(s_ee,lattice_metric)*(1.0-jnp.eye(n))

		if include_r_ae:
			ae_features=jnp.concatenate((r_ae[...,None],ae),axis=2)
		else:
			ae_features=ae
		ae_features=jnp.reshape(ae_features,[jnp.shape(ae_features)[0],-1])
		ee_features=jnp.concatenate((r_ee[...,None],ee),axis=2)
		return ae_features,ee_features

	return networks.FeatureLayer(init=init,apply=apply)


def make_ferminet_decaying_features(
		charges: Optional[jnp.ndarray] = None,
		nspins: Optional[Tuple[int,...]] = None,
		ndim: int = 3,
		rc: float = 3.0,
		rc_smth: float = 0.5,
		lr: Optional[bool] = False,
		lattice: Optional[jnp.ndarray] = None,
) -> networks.FeatureLayer:
	"""Returns the init and apply functions for the decaying features."""

	if lr:
		pbc_feat=make_pbc_feature_layer(
			charges,nspins,ndim,
			lattice=lattice,
			include_r_ae=True,
		)

	def init() -> Tuple[Tuple[int,int],networks.Param]:
		if lr:
			return (pbc_feat.init()[0][0]+ndim+1,pbc_feat.init()[0][1]+ndim+1),{}
		else:
			return (ndim+1,ndim+1),{}

	def make_pref(rr):
		"""
				  sw(r)
		pref = -----------
				 (r+1)^2
		"""
		sw=dp.switch_func_poly(rr,rc,rc_smth)
		return sw/((rr+1.0)*(rr+1.0))

	def apply(ae,r_ae,ee,r_ee) -> Tuple[jnp.ndarray,jnp.ndarray]:
		ae_pref=make_pref(r_ae)
		ee_pref=make_pref(r_ee)
		ae_features=jnp.concatenate((r_ae,ae),axis=2)
		ee_features=jnp.concatenate((r_ee,ee),axis=2)
		ae_features=ae_pref*ae_features
		ee_features=ee_pref*ee_features
		ae_features=jnp.reshape(ae_features,[jnp.shape(ae_features)[0],-1])
		if lr:
			ae_pbc_feat,ee_pbc_feat=pbc_feat.apply(ae,r_ae,ee,r_ee)
			ae_features=jnp.concatenate([ae_features,ae_pbc_feat],axis=-1)
			ee_features=jnp.concatenate([ee_features,ee_pbc_feat],axis=-1)
		return ae_features,ee_features

	return networks.FeatureLayer(init=init,apply=apply)


def make_open_features(
		charges: Optional[jnp.ndarray] = None,
		nspins: Optional[Tuple[int,...]] = None,
		ndim: int = 3,
		feature_scale:bool=False,
		feature_scale_num:Tuple[int,...]=(1),
		has_decay: Optional[bool] = False,
):

	def init() -> Tuple[Tuple[int,int],networks.Param]:
		if feature_scale:
			dim_one_feature=(ndim+1)*len(feature_scale_num)
			dim_two_feature=(ndim+1)*len(feature_scale_num)
			if has_decay:
				dim0+=1  #5
				dim1+=1
		else:
			dim_one_feature=ndim+1 
			dim_two_feature=ndim+1
			if has_decay:
				dim0+=1  #5
				dim1+=1
		return (dim_one_feature,dim_two_feature),{}

	def apply_(ae,r_ae,ee,r_ee,aa=None,r_aa=None) -> Tuple[jnp.ndarray,jnp.ndarray]:
		# different ee convention, so use -ee
		nele=ee.shape[0]
		# print("ee.shape:",ee.shape)   #(nele,nele,3)
		ee=-ee*(1-jnp.eye(nele))[...,None]
		r_ee=-r_ee*(1-jnp.eye(nele))[...,None]

		ae_features=jnp.concatenate([ae,r_ae],axis=-1)
		ee_features=jnp.concatenate([ee,r_ee],axis=-1)
		
		if aa is not None:
			na=aa.shape[0]
			aa=-aa*(1-jnp.eye(na))[...,None]
			r_aa=-r_aa*(1-jnp.eye(na))[...,None]
			aa_features=jnp.concatenate([aa,r_aa],axis=-1)
		else:
			aa_features=None

		if has_decay:
			def add_decay_feat(features,r_ae):
				features=jnp.concatenate([features,decay.deacy(r_ae)],axis=-1)
				return features
			ae_features=add_decay_feat(ae_features,r_ae)
			ee_features=add_decay_feat(ee_features,r_ee)
			if aa is not None:
				aa_features=add_decay_feat(aa_features,r_aa)
		#print('ee_features:',aa_features[0])
		
		if feature_scale:
			ae_features_scale=[]
			ee_features_sacle=[]
			aa_features_scale=[]
			for num in feature_scale_num:
				ae_features_scale+=[ae_features/float(num)]
				ee_features_sacle+=[ee_features/float(num)]
				aa_features_scale+=[aa_features/float(num)]
			ae_features_scale=jnp.concatenate(ae_features_scale,axis=-1)
			ee_features_sacle=jnp.concatenate(ee_features_sacle,axis=-1)
			aa_features_scale=jnp.concatenate(aa_features_scale,axis=-1)
			return ae_features_scale,ee_features_sacle,aa_features_scale
		else:
			return ae_features,ee_features,aa_features

	return networks.FeatureLayer(init=init,apply=apply_)



def make_ds_features_ef(
		charges: Optional[jnp.ndarray] = None,
		nspins: Optional[Tuple[int,...]] = None,
		ndim: int = 3,
		lattice: Optional[jnp.ndarray] = None,
		has_ds: bool = True,
		has_sym: bool = False,
		has_cos: bool = False,
		has_sin: bool = False,
		has_pbc_norm: bool = False,
		scale: Union[float,List[float]] = [],  #scale既可以被赋值为一个浮点数，也可以被赋值为一个包含浮点数的列表。
		numb_divid: int = 1,
		do_act: bool = False,
		act_func: str = 'tanh',
):
	if type(scale) is float:
		scale=[scale]
	all_scales=[1.0]+scale+[1./(ii+1.) for ii in range(numb_divid)]
	# sorted unique list
	all_scales=sorted(list(set([float(ii) for ii in all_scales])))
	#set([float(ii) for ii in all_scales])：通过将列表转换为集合，自动去除了重复的元素。集合是一个无序的数据结构，不允许有重复的值。
	#这一步操作将去除了列表中可能存在的任何重复的浮点数。
	#由于集合是无序的，为了能对元素进行排序，需要再次将集合转换回列表。
	#最后，使用sorted函数对去重并转换为列表的结果进行排序。sorted函数返回一个新的列表，其中的元素按升序排列
	num_scales=len(all_scales)
	if act_func=='tanh':
		act_func=jnp.tanh
	elif act_func=='tanh2':
		act_func=lambda x:2.*jnp.tanh(x/2.)
	elif act_func=='tanh3':
		act_func=lambda x:3.*jnp.tanh(x/3.)
	else:
		raise RuntimeError(f'unknow act func {act_func}')

	if lattice is not None:
		org_lattice=lattice/(2.*jnp.pi)
		rec_lattice=jnp.linalg.inv(org_lattice)
		inv_lattice=jnp.linalg.inv(lattice)
		lattice_metric=lattice.T@lattice
	else:
		org_lattice=None
		rec_lattice=None
		inv_lattice=None

	def init() -> Tuple[Tuple[int,int],networks.Param]:
		dim0,dim1=0,0
		if has_ds:
			dim0+=(ndim+1)*num_scales
			dim1+=(ndim+1)*num_scales
		if has_cos:
			dim0+=ndim
			dim1+=ndim
		if has_sin:
			dim0+=ndim
			dim1+=ndim
		if has_pbc_norm:
			dim0+=1
			dim1+=1
		if has_sym:
			dim0+=ndim*num_scales
			dim1+=ndim*num_scales
		return (dim0,dim1),{}

	def apply_(ee,r_ee,) -> Tuple[jnp.ndarray,jnp.ndarray]:
		# different ee convention, so use -ee
		n=ee.shape[0]
		ee_features_list=[]

		# ds_features
		if has_ds:
			sim_periodic_see,sim_periodic_xee=ds.nu_distance(
				-ee+jnp.eye(n)[...,None],org_lattice,rec_lattice,has_sym=has_sym)
			sim_periodic_see=sim_periodic_see*(1.0-jnp.eye(n))
			sim_periodic_see=sim_periodic_see[...,None]
			sim_periodic_xee=sim_periodic_xee*(1.0-jnp.eye(n))[...,None]
			ee_features_=jnp.concatenate([sim_periodic_see,sim_periodic_xee],axis=-1)
			ee_features_list=[ee_features_*ss for ss in all_scales]
			ee_features_list=[act_func(ee) if do_act else ee for ee in ee_features_list]
		# sin-cos features
		if has_cos:
			def add_cos_feat(_ae):
				s_ae=jnp.matmul(_ae,inv_lattice)
				cos__ae_feat=jnp.cos(2.*jnp.pi*s_ae)
				return cos__ae_feat

			ee_features_list.append(add_cos_feat(ee))
		if has_sin:
			def add_sin_feat(_ae):
				s_ae=jnp.matmul(_ae,inv_lattice)
				sin__ae_feat=jnp.sin(2.*jnp.pi*s_ae)
				return sin__ae_feat

			ee_features_list.append(add_sin_feat(ee))
		if has_pbc_norm:
			def periodic_norm(vec,metric):
				a=(1-jnp.cos(2*jnp.pi*vec))
				b=jnp.sin(2*jnp.pi*vec)
				# i,j = nelectron, natom for ae
				cos_term=jnp.einsum('ijm,mn,ijn->ij',a,metric,a)
				sin_term=jnp.einsum('ijm,mn,ijn->ij',b,metric,b)
				return (1/(2*jnp.pi))*jnp.sqrt(cos_term+sin_term)

			ee_features_list.append(
				(periodic_norm(ee,lattice_metric)*(1.0-jnp.eye(n)))[...,None]
			)

		ee_features=jnp.concatenate(ee_features_list,axis=-1)

		return ee_features

	return networks.FeatureLayer(init=init,apply=apply_)

