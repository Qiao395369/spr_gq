from typing import Optional, Sequence, Tuple, Union
import vmcnet.gaoqiao.envelopes as envelopes
import vmcnet.gaoqiao.networks as networks
import vmcnet.gaoqiao.dp as dp
import vmcnet.gaoqiao.open_feature_layer as open_feature_layer
import vmcnet.gaoqiao.jastrows as jastrows
import jax
import ml_collections
import logging
def build_network(
	n, charges, 
	nspins: Tuple[int, ...],
	key, 
	ndet, depth, h1, h2, nh, do_complex,
	gq_type:str= 'ef',
	envelope_type:str= 'ds_hz',
	layer_update_scheme: Optional[dict] = None,
	attn: Optional[dict] = None,
	trimul: Optional[dict] = None,
	h1_attn: Optional[dict] = None,
	feat_params: Optional[dict] = None,
	det_mode: str = "det",
	gemi_params: Optional[dict] = None,
	jastrow_type: str = "mlp",
	jastrow_mlp_nlayer: int = 4,
	jastrow_mlp_ndim: int = 64,
	RHF: bool = False,
	activation_type: str = "tanh",
):
	if activation_type == "tanh":
		activation_fn = jax.nn.tanh
	elif activation_type == "relu":
		activation_fn = jax.nn.relu
	elif activation_type == "silu":
		activation_fn = jax.nn.silu

	hidden_dims=tuple([(h1, h2) for _ in range(depth)])
	ndim = 3 
	natom=len(charges)
	dim_extra_params=0
	use_last_layer=False
	full_det=True
	do_aa=True
	bias_orbitals=False
	reduced_h1_size=None
	gemi_params=None
	full_det=True
	hf_solution=None
	make_envelope_kwargs = {"hiddens": [] if nh==0 else [nh],}
	mes = dp.ManyElectronSystem(charges, nspins)
	make_feature_layer_kwargs={}
	for kk, vv in feat_params.items():
		make_feature_layer_kwargs[kk] = vv

	#build feature_layer : pp,r_pp --> h2 features
	feature_layer = open_feature_layer.make_open_features_ef(  
			n,
			ndim,
			**make_feature_layer_kwargs
		)  # type: networks.FeatureLayer
	
	if envelope_type=="ds_hz":
		envelope = envelopes.make_ds_hz_envelope(**make_envelope_kwargs)  # type: envelopes.Envelope
	elif envelope_type=="iso":
		envelope = envelopes.make_isotropic_envelope()
	else :
		raise ValueError("envelope_type should be in ['ds_hz', 'iso']")
	
	if jastrow_type in ["mlp","mlp_res"]:
		jastrow = jastrows.make_mlp_jastrow(
			nspins = nspins,
			hiddenlayers_num=jastrow_mlp_nlayer,
			hiddenlayers_size=jastrow_mlp_ndim,
			activation_fn=activation_fn,
			residual=True if jastrow_type=="mlp_res" else False,
			)
	elif jastrow_type == "simple_ee":
		jastrow = jastrows.make_simple_ee_jastrow(
			nspins = nspins,
			)
	elif jastrow_type == "null":
		jastrow = jastrows.make_null_jastrow()
	else:
		raise ValueError("jastrow_type should be in ['mlp','mlp_res','simple_ee','null]")

	#build ferminet_model : h2(0) features --> h1(L) 
	logging.info("wfn type: %s ", (gq_type))
	if gq_type == "ef":
		ef=True
		ferminet_model = networks.make_fermi_net_model_ef(  
			n, 
			ndim,
			nspins,
			feature_layer,
			hidden_dims,
			use_last_layer,
			dim_extra_params=dim_extra_params,
			do_aa=do_aa,
			mes=mes,
			layer_update_scheme=layer_update_scheme,
			attn_params=attn,
			trimul_params=trimul,
			reduced_h1_size=reduced_h1_size,
			h1_attn_params=h1_attn,
		)

	elif gq_type == "ef_test":
		ef=True
		ferminet_model = networks.make_fermi_net_model_ef_test(
			n, 
			ndim,
			nspins,
			feature_layer,
			hidden_dims,
			use_last_layer,
			dim_extra_params=dim_extra_params,
			do_aa=do_aa,
			mes=mes,
			layer_update_scheme=layer_update_scheme,
			attn_params=attn,
			trimul_params=trimul,
			reduced_h1_size=reduced_h1_size,
			h1_attn_params=h1_attn,
		)
	elif gq_type == "ef_shrd":
		ef=True
		ferminet_model = networks.make_fermi_net_model_ef_shrd(
			n, 
			ndim,
			nspins,
			feature_layer,
			hidden_dims,
			use_last_layer,
			dim_extra_params=dim_extra_params,
			do_aa=do_aa,
			mes=mes,
			activation_fn=activation_fn,
			layer_update_scheme=layer_update_scheme,
			attn_params=attn,
			trimul_params=trimul,
			reduced_h1_size=reduced_h1_size,
			h1_attn_params=h1_attn,
		)
	elif gq_type == "ef_shrd_sym":
		ef=True
		ferminet_model = networks.make_fermi_net_model_ef_shrd_sym(
			n, 
			ndim,
			nspins,
			feature_layer,
			hidden_dims,
			use_last_layer,
			dim_extra_params=dim_extra_params,
			do_aa=do_aa,
			mes=mes,
			activation_fn=activation_fn,
			attn_params=attn,
		)

	elif gq_type == "fermi":
		ef=False
		envelope = envelopes.make_isotropic_envelope()
		feature_layer = networks.make_ferminet_features()
		ferminet_model = networks.make_fermi_net_model(
			natom,
			nspins,
			feature_layer,
			hidden_dims,
			use_last_layer,
			dim_extra_params=dim_extra_params,
			do_aa=do_aa,
			mes=mes,
		)

	elif gq_type == "shrd":
		ef=False
		feature_layer = open_feature_layer.make_open_features()
		ferminet_model = networks.make_fermi_net_model_zinv_shrd(
			natom, 
			ndim,
			nspins,
			feature_layer,
			hidden_dims,
			use_last_layer,
			dim_extra_params=dim_extra_params,
			do_aa=do_aa,
			mes=mes,
			distinguish_ele=True,
			code_only_first=True,
			attn_params=None,
			attn_1_params=None,
		)

	network_init, signed_network, det_fn, network_options, orbitals = networks.make_fermi_net(
		n, 
		ndim, 
		nspins,
		charges,
		envelope=envelope,
		feature_layer=feature_layer,
		ferminet_model=ferminet_model,
		bias_orbitals=bias_orbitals,
		use_last_layer=use_last_layer,
		hf_solution=hf_solution,
		full_det=full_det,
		hidden_dims=hidden_dims,
		determinants=ndet,
		do_complex=do_complex,
		do_aa=do_aa,
		mes=mes,
		det_mode=det_mode,
		gemi_params=gemi_params,
		equal_footing=ef,
		gq_type=gq_type,
		jastrow=jastrow,
		RHF=RHF,
	)
  
	key, subkey = jax.random.split(key)
	params = network_init(subkey)

	return params, signed_network, det_fn, orbitals

if __name__=='__main__':
	n = 14
	key = jax.random.PRNGKey(42)
	nk = 7
	ndet = 1
	depth = 4
	h1 = 16
	h2 = 16
	nh = 8 

	params, network = build_network(n, key, nk, ndet, depth, h1, h2, nh)

