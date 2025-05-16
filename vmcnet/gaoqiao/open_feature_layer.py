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

def make_open_features(
		# charges: Optional[jnp.ndarray] = None,
		# nspins: Optional[Tuple[int,...]] = None,
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



def make_open_features_ef(
		# charges: Optional[jnp.ndarray] = None,
		# nspins: Optional[Tuple[int,...]] = None,
		ndim: int = 3,
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


	def init() -> Tuple[Tuple[int,int],networks.Param]:
		dim0,dim1=0,0
		dim0+=(ndim+1)*num_scales
		dim1+=(ndim+1)*num_scales
		return (dim0,dim1),{}

	def apply_(ee,r_ee,) -> Tuple[jnp.ndarray,jnp.ndarray]:
		# different ee convention, so use -ee
		n=ee.shape[0]
		ee_features_list=[]
		ee=-ee*(1.0-jnp.eye(n))[...,None]
		r_ee=-r_ee*(1-jnp.eye(n))[...,None]

		ee_features_=jnp.concatenate([ee,r_ee],axis=-1)
		ee_features_list=[ee_features_*ss for ss in all_scales]
		ee_features_list=[act_func(ee) if do_act else ee for ee in ee_features_list]

		ee_features=jnp.concatenate(ee_features_list,axis=-1)

		return ee_features

	return networks.FeatureLayer(init=init,apply=apply_)

