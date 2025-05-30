from typing import Optional, Sequence, Tuple, Union
import vmcnet.gaoqiao.envelopes as envelopes
import vmcnet.gaoqiao.networks as networks
import vmcnet.gaoqiao.dp as dp
import vmcnet.gaoqiao.open_feature_layer as open_feature_layer
import jax
import ml_collections

def build_network(
	n, charges, nspins,
	key, 
	ndet, depth, h1, h2, nh, do_complex,
	gq_type:str= 'ef',
	ef_construct_features_type:str= 'conv_0',
	envelope_type:str= 'ds_hz',
	ef: bool = False,
	layer_update_scheme: Optional[dict] = None,
	attn: Optional[dict] = None,
	trimul: Optional[dict] = None,
	h1_attn: Optional[dict] = None,
	feat_params: Optional[dict] = None,
	det_mode: str = "det",
	gemi_params: Optional[dict] = None,
):

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
			ndim,
			**make_feature_layer_kwargs
		)  # type: networks.FeatureLayer
	
	if envelope_type=="ds_hz":
			envelope = envelopes.make_ds_hz_envelope(**make_envelope_kwargs)  # type: envelopes.Envelope
	elif envelope_type=="iso":
		envelope = envelopes.make_isotropic_envelope()
	else :
		raise ValueError("envelope_type should be in ['ds_hz', 'iso']")
	
	#build ferminet_model : h2(0) features --> h1(L) 
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
			ef_construct_features_type=ef_construct_features_type,
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

	network_init, signed_network, network_options = networks.make_fermi_net(
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
	)
  
	key, subkey = jax.random.split(key)
	params = network_init(subkey)

	return params, signed_network

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

