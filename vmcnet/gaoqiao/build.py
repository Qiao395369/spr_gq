# Author:Qiaoqiao
# Date:2024/3/20
# filename:build
# Description:

import functools
import importlib
import time
from typing import Optional,Sequence,Tuple,Union

import vmcnet.gaoqiao.envelopes as envelopes
import vmcnet.gaoqiao.networks as networks
import vmcnet.gaoqiao.dp as dp
import vmcnet.gaoqiao.open_feature_layer as open_feature_layer
import vmcnet.gaoqiao.envelopes as envelopes
import jax
import jax.numpy as jnp
import numpy as np

import ml_collections

#cfg=get_config(nk,ndet,depth,h1,h2,nh, ef=ef,feat_params=feat_params,attn=attn,trimul=trimul,h1_attn=h1_attn,
# det_mode=det_mode,gemi_params=gemi_params,)
def get_config(
		nk,ndet,depth,h1,h2,nh,do_complex,
		ef: bool = False,
		attn: Optional[dict] = None,
		trimul: Optional[dict] = None,
		h1_attn: Optional[dict] = None,
		feat_params: Optional[dict] = None,
		det_mode: str = "det",
		gemi_params: Optional[dict] = None,
):
	if attn is not None or trimul is not None or h1_attn is not None:
		if not ef:
			raise RuntimeError(
				"attn, h1_attn and trimul is only supported with ef"
			)

	cfg=ml_collections.ConfigDict({'network':{'detnet':{}}})


	cfg.network.make_feature_layer_fn=(
			"open_feature_layer.make_open_features")
	cfg.network.make_feature_layer_kwargs={
		"has_decay":False,
	}

	if ef and feat_params is not None:
		# only valid for ef
		for kk,vv in feat_params.items():
			cfg.network.make_feature_layer_kwargs[kk]=vv

	cfg.network.make_envelope_fn=(
		"envelopes.make_ds_hz_envelope")
	cfg.network.make_envelope_kwargs={
		"hiddens":[] if nh==0 else [nh],  #nh:twist hidden size,default=8
	}

	if not ef:
		cfg.network.make_model_fn=(
			"networks.make_fermi_net_model_zinv_shrd")
		cfg.network.make_model_kwargs={
			"distinguish_ele":True,
			"code_only_first":True,
		}
	else:
		cfg.network.make_model_fn=(
			"networks.make_fermi_net_model_ef")
		cfg.network.make_model_kwargs={}
		if attn is not None:
			cfg.network.make_model_kwargs['attn_params']=attn
		if trimul is not None:
			cfg.network.make_model_kwargs['trimul_params']=trimul
		if h1_attn is not None:
			cfg.network.make_model_kwargs['h1_attn_params']=h1_attn

	if det_mode=="det":
		cfg.network.full_det=True
		cfg.network.detnet.orb_numb_k=nk  #1
		cfg.network.detnet.env_twist_hiddens=[]
		gemi_params=None
	elif det_mode=="gemi":
		cfg.network.full_det=False

	else:
		raise RuntimeError(f"unknown det_mode {det_mode}")

	cfg.network.use_last_layer=False
	cfg.network.bias_orbitals=False
	cfg.network.detnet.det_mode=det_mode
	cfg.network.detnet.gemi_params=gemi_params

	cfg.network.detnet.determinants=ndet
	cfg.network.detnet.hidden_dims=tuple([(h1,h2) for _ in range(depth)])

	cfg.network.detnet.do_complex=do_complex
	cfg.network.detnet.do_twist=False
	cfg.network.detnet.numb_k=0
	cfg.network.detnet.do_aa=True

	return cfg


def build_network(
		n,charges,nspins,key,nk,ndet,depth,h1,h2,nh,do_complex,
		feature_scale:bool=False,
		ef: bool = False,
		attn: Optional[dict] = None,
		trimul: Optional[dict] = None,
		h1_attn: Optional[dict] = None,
		feat_params: Optional[dict] = None,
		det_mode: str = "det",
		gemi_params: Optional[dict] = None,
):
	"""charges:array that contains the charge of each atom."""
	ndim=3
	# nspins=(n//2,n//2)
	#ef=False
	#print(ef)
	# charges=jnp.ones(natom)
	# print("charge.shape[0]:",charges.shape[0])
	# print("sum(nspins):",sum(nspins))
	mes=dp.ManyElectronSystem(charges,nspins)

	cfg=get_config(nk,ndet,depth,h1,h2,nh,do_complex,
	               ef=ef,feat_params=feat_params,
	               attn=attn,trimul=trimul,h1_attn=h1_attn,
	               det_mode=det_mode,gemi_params=gemi_params,
	               )

	# feature_layer_module,feature_layer_fn=(
	# 	cfg.network.make_feature_layer_fn.rsplit('.',maxsplit=1))
	# feature_layer_module=importlib.import_module(feature_layer_module)
	# make_feature_layer=getattr(feature_layer_module,feature_layer_fn)
	feature_layer=open_feature_layer.make_open_features(
		charges,
		nspins,
		ndim,
		feature_scale,
		**cfg.network.make_feature_layer_kwargs)  # type: networks.FeatureLayer

	# envelope_module,envelope_fn=(
	# 	cfg.network.make_envelope_fn.rsplit('.',maxsplit=1))
	# envelope_module=importlib.import_module(envelope_module)
	# make_envelope=getattr(envelope_module,envelope_fn)
	envelope=envelopes.make_ds_hz_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope

	# model_module,model_fn=(
	# 	cfg.network.make_model_fn.rsplit('.',maxsplit=1))
	# model_module=importlib.import_module(model_module)
	# make_model=getattr(model_module,model_fn)
	ferminet_model=networks.make_fermi_net_model_zinv_shrd(
		n,
		ndim,
		nspins,
		feature_layer,
		cfg.network.detnet.hidden_dims,
		cfg.network.use_last_layer,
		dim_extra_params=0,
		do_aa=cfg.network.detnet.do_aa,
		mes=mes,
		**cfg.network.make_model_kwargs,
	)

	network_init,signed_network,network_options,orbitals=networks.make_fermi_net(
		n,
		ndim,
		nspins,
		charges,
		envelope=envelope,
		feature_layer=feature_layer,
		ferminet_model=ferminet_model,
		bias_orbitals=cfg.network.bias_orbitals,
		use_last_layer=cfg.network.use_last_layer,
		hf_solution=None,
		full_det=cfg.network.full_det,
		mes=mes,
		equal_footing=ef,
		**cfg.network.detnet,
	)

	key,subkey=jax.random.split(key)
	params=network_init(subkey)

	return params,signed_network,orbitals


if __name__=='__main__':
	n=14
	L=1.235
	key=jax.random.PRNGKey(42)
	nk=7
	ndet=1
	depth=4
	h1=16
	h2=16
	nh=8

	params,network=build_network(n,L,key,nk,ndet,depth,h1,h2,nh)

