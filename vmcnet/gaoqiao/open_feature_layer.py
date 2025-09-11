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
import logging
import vmcnet.gaoqiao.networks as networks
import vmcnet.gaoqiao.dp as dp

def make_open_features(charges: Optional[jnp.ndarray] = None,
                           nspins: Optional[Tuple[int, ...]] = None,
                           ndim: int = 3) :
	del charges, nspins
	def init() -> Tuple[Tuple[int,int],networks.Param]:
		dim0,dim1=0,0
		dim0+=ndim+1
		dim1+=ndim+1
		return (dim0,dim1),{}

	def apply(ae, r_ae,ea,r_ea, ee, r_ee, aa=None, r_aa=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
		# different ee convention, so use -ee
		ne=ee.shape[0]
		na=aa.shape[0]
		ee=-ee*(1.0-jnp.eye(ne))[...,None]
		r_ee=-r_ee*(1-jnp.eye(ne))[...,None]

		aa=-aa*(1.0-jnp.eye(na))[...,None]
		r_aa=-r_aa*(1-jnp.eye(na))[...,None]

		ee_features=jnp.concatenate([ee,r_ee],axis=-1)
		aa_features=jnp.concatenate([aa,r_aa],axis=-1)
		ae_features=jnp.concatenate([ae,-r_ae],axis=-1)
		ea_features=jnp.concatenate([ea,-r_ea],axis=-1)

		return ae_features,ea_features, ee_features, aa_features

	return networks.FeatureLayer(init=init,apply=apply)


def make_open_features_ef(
		# charges: Optional[jnp.ndarray] = None,
		# nspins: Optional[Tuple[int,...]] = None,
		nele:int, 
		ndim: int = 3,
		scale: Union[float,List[float]] = [],  #scale既可以被赋值为一个浮点数，也可以被赋值为一个包含浮点数的列表。
		numb_divid: int = 1,
		do_act: bool = False,
		act_func: str = 'tanh',
		rescale:str="all",
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




	def init() -> Tuple[Tuple[int,int],networks.Param]:
		dim0,dim1=0,0
		dim0+=(ndim+1)*num_scales
		dim1+=(ndim+1)*num_scales
		return (dim0,dim1),{}

	def apply_(pp,r_pp,) -> Tuple[jnp.ndarray,jnp.ndarray]:
		# different ee convention, so use -ee
		n=pp.shape[0]
		pp_features_list=[]
		pp=-pp*(1.0-jnp.eye(n))[...,None]

		r_pp=r_pp*(1.0-jnp.eye(n))[...,None]
		if rescale=="all_eps":
			eps=1e-5
			log_r_pp = jnp.log(1 + r_pp)  # grows as log(r) rather than r
			pp_features_ = jnp.concatenate((pp * log_r_pp / (r_pp+eps), log_r_pp ), axis=2)
		elif rescale=="all":
			log_r_pp = jnp.log(1 + r_pp)  
			factor=jnp.where(log_r_pp!=0, log_r_pp / r_pp, 0.0)
			pp_features_ = jnp.concatenate(( pp * factor , log_r_pp ), axis=2)
		elif rescale=="e-a":
			ee,ea,ae,aa=split_ee_ea_ae_aa(pp,nele)
			r_ee,r_ea,r_ae,r_aa=split_ee_ea_ae_aa(r_pp,nele)
			ee_features_=jnp.concatenate([ee,r_ee],axis=-1)

			log_r_ea = jnp.log(1 + r_ea)  
			ea_features_ = jnp.concatenate((ea * log_r_ea /r_ea , log_r_ea ), axis=-1)

			log_r_ae = jnp.log(1 + r_ae)  
			ae_features_ = jnp.concatenate((ae * log_r_ae /r_ae , log_r_ae ), axis=-1)

			aa_features_=jnp.concatenate([aa,r_aa],axis=-1)

			pp_features_ = reform_ee_ea_ae_aa(ee_features_,ea_features_,ae_features_,aa_features_)
		elif rescale=="None":
			pp_features_=jnp.concatenate([pp,r_pp],axis=-1)
		else:
			raise RuntimeError(f"unknow rescale {rescale}")
			
		pp_features_list=[pp_features_*ss for ss in all_scales]
		pp_features_list=[act_func(pp) if do_act else pp for pp in pp_features_list]

		pp_features=jnp.concatenate(pp_features_list,axis=-1)
		logging.info("pp_feature: %s", pp_features.shape)
		return pp_features

	return networks.FeatureLayer(init=init,apply=apply_)


def split_ee_ea_ae_aa(data, ne):
	"""
	data : np x np x ...
	"""
	ea_split = [ne]  #[nele]
	split0 = jnp.split(data, ea_split, axis=0)  #(np,np,...)->(nele,np,...),(na,np,...)
	[ee, ea] = jnp.split(split0[0], ea_split, axis=1)  #(nele,np,...)->(nele,nele,...),(nele,na,...)
	[ae, aa] = jnp.split(split0[1], ea_split, axis=1)  #(na,np,...)->(na,nele,...),(na,na,...)
	return ee, ea, ae, aa
  
def reform_ee_ea_ae_aa(ee, ea, ae, aa):
	ee_ea=jnp.concatenate([ee,ea],axis=1)
	ae_aa=jnp.concatenate([ae,aa],axis=1)
	return jnp.concatenate([ee_ea,ae_aa],axis=0)