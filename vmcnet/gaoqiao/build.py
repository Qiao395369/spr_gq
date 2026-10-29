import dataclasses
from typing import Optional, Tuple, Union
import vmcnet.gaoqiao.envelopes as envelopes
import vmcnet.gaoqiao.networks as networks
import vmcnet.gaoqiao.so3_utils as so3_utils
import vmcnet.gaoqiao.so3_l0 as so3_l0
import vmcnet.gaoqiao.dp as dp
import vmcnet.gaoqiao.open_feature_layer as open_feature_layer
import vmcnet.gaoqiao.jastrows as jastrows
import jax
import logging

def _get_activation(name: str):
    if name == "tanh":
        return jax.nn.tanh
    if name == "relu":
        return jax.nn.relu
    if name == "silu":
        return jax.nn.silu
    raise ValueError(
        f"Unsupported activation {name!r}; expected one of ['tanh', 'relu', 'silu']."
    )


def _make_so3_config(
    so3_params: Optional[Union[dict, so3_utils.SO3Config]],
    dtype: Optional[object] = None,
) -> so3_utils.SO3Config:
    """构造 SO(3) 配置；顶层 dtype 指定时，以它为准。"""
    if so3_params is None:
        config = so3_utils.SO3Config()
    elif isinstance(so3_params, so3_utils.SO3Config):
        config = so3_params
    elif isinstance(so3_params, dict):
        config = so3_utils.SO3Config(**so3_params)
    else:
        raise TypeError("so3_params must be None, dict, or so3_utils.SO3Config")
    return config if dtype is None else dataclasses.replace(config, float_dtype=dtype)


def build_network(
    n,
    charges,
    nspins: Tuple[int, ...],
    key,
    ndet,
    depth,
    h1,
    h2,
    nh,
    do_complex,
    gq_type: str = "ef",
    envelope_type: str = "ds_hz",
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
    jastrow_activation_type: str = "tanh",
    dp_type: str = "element1",
    alpha: float = 1.0,
    # NEW: used only when gq_type == "so3".
    # May be a dict accepted by so3_utils.SO3Config, an SO3Config object,
    # or None to use SO3Config defaults.
    so3_params: Optional[Union[dict, so3_utils.SO3Config]] = None,
    dtype: Optional[object] = None,
    so3_l0_params: Optional[Union[dict, so3_l0.SO3L0Config]] = None,
):
    # 独立入口提前返回，不构造普通网络的 feature/envelope/MLP Jastrow。
    if gq_type == "so3_l0":
        if det_mode != "det":
            raise ValueError('gq_type="so3_l0" 要求 det_mode="det"。')
        if n != sum(nspins):
            raise ValueError("so3_l0 的 n 必须等于 nspins 之和。")
        if RHF:
            logging.warning("RHF=True 被 so3_l0 忽略；两组自旋使用独立轨道头。")
        if so3_l0_params is None:
            config = so3_l0.SO3L0Config()
        elif isinstance(so3_l0_params, so3_l0.SO3L0Config):
            config = so3_l0_params
        elif isinstance(so3_l0_params, dict):
            config = so3_l0.SO3L0Config(**so3_l0_params)
        else:
            raise TypeError("so3_l0_params 必须为 dict、SO3L0Config 或 None。")
        if dtype is not None:
            config = dataclasses.replace(config, float_dtype=dtype)
        mes = dp.ManyElectronSystem(charges, nspins, dp_type)
        logging.info("wfn type: so3_l0; ndet=%s; config=%s", ndet, config)
        _, model_key = jax.random.split(key)
        return so3_l0.make_network(model_key, charges, nspins, ndet,
                                    mes.get_part_features(), config)

    # 同一个 so3 主干的读出消融；无 A/rank，绕开下面的旧 sector 组装。
    if gq_type == "so3":
        so3_config = _make_so3_config(so3_params, dtype=dtype)
        if so3_utils.readout_mode(so3_config) == "mlp":
            if det_mode != "det":
                raise ValueError('SO(3) MLP 读出要求 det_mode="det"。')
            if n != sum(nspins):
                raise ValueError("SO(3) MLP 读出的 n 必须等于 nspins 之和。")
            if RHF:
                logging.warning("RHF=True 被 SO(3) MLP 读出忽略；两组自旋使用独立输出头。")
            mes = dp.ManyElectronSystem(charges, nspins, dp_type)
            logging.info("wfn type: so3; MLP readout; ndet=%s; config=%s", ndet, so3_config)
            _, model_key = jax.random.split(key)
            return so3_utils.make_mlp_network(
                model_key, charges, nspins, ndet, mes.get_part_features(), so3_config)

    activation_fn = _get_activation(activation_type)
    jastrow_activation_fn = _get_activation(jastrow_activation_type)

    hidden_dims = tuple((h1, h2) for _ in range(depth))
    ndim = 3
    natom = len(charges)
    dim_extra_params = 0
    use_last_layer = False
    full_det = True
    do_aa = True
    bias_orbitals = False
    reduced_h1_size = None
    gemi_params=None
    hf_solution = None
    make_envelope_kwargs = {"hiddens": [] if nh == 0 else [nh]}
    mes = dp.ManyElectronSystem(charges, nspins, dp_type)
    feat_params = {} if feat_params is None else dict(feat_params)


    feature_layer = open_feature_layer.make_open_features_ef(
        n,
        ndim,
        **feat_params,
    )
    if envelope_type == "ds_hz":
        envelope = envelopes.make_ds_hz_envelope(**make_envelope_kwargs,activation_fn=activation_fn)
    elif envelope_type == "ds_hz_old":
        envelope = envelopes.make_ds_hz_envelope_old(**make_envelope_kwargs,activation_fn=activation_fn)
    elif envelope_type == "ds_hz_deg":
        envelope = envelopes.make_ds_hz_degenerate_envelope(**make_envelope_kwargs,activation_fn=activation_fn)
    elif envelope_type == "iso":
        envelope = envelopes.make_isotropic_envelope()
    elif envelope_type == "iso_new":
        envelope = envelopes.make_isotropic_envelope_new()
    else:
        raise ValueError("envelope_type should be in ['ds_hz', 'ds_hz_old', 'iso', 'iso_new']")

    if jastrow_type in ["mlp", "mlp_res"]:
        jastrow = jastrows.make_mlp_jastrow(
            nspins=nspins,
            hiddenlayers_num=jastrow_mlp_nlayer,
            hiddenlayers_size=jastrow_mlp_ndim,
            activation_fn=jastrow_activation_fn,
            residual=(jastrow_type == "mlp_res"),
        )
    elif jastrow_type == "simple_ee":
        jastrow = jastrows.make_simple_ee_jastrow(nspins=nspins)
    elif jastrow_type == "null":
        jastrow = jastrows.make_null_jastrow()
    else:
        raise ValueError("jastrow_type should be in ['mlp', 'mlp_res', 'simple_ee', 'null']")

    # ==================================================================
    # NEW SO(3) BRANCH
    # ==================================================================
    if gq_type == "so3":
        # SO(3) 的数域由 so3_params["basis"] 决定；旧 do_complex 标志
        # 只属于普通 FermiNet，不应阻止复基 SO(3) 模式的构建。
        if det_mode != "det":
            raise ValueError(
                'gq_type="so3" currently requires det_mode="det".'
            )
        if RHF:
            logging.warning(
                "RHF=True is ignored by the current SO(3) determinant head."
            )

        so3_config = _make_so3_config(so3_params, dtype=dtype)
        # Performs dtype validation. In particular float64 requires JAX x64.
        so3_utils.config_dtype(so3_config)
        so3_utils.sector_list(so3_config)

        logging.info("wfn type: so3; config=%s", so3_config)

        # Important: SO(3) does NOT use the old envelope, feature layer, or
        # FermiNet backbone. make_fermi_net() accepts these as None when
        # so3_config is provided.
        network_init, signed_network, det_fn, network_options, orbitals, hz = (
            networks.make_fermi_net(
                n,
                ndim,
                nspins,
                charges,
                envelope=None,
                feature_layer=None,
                ferminet_model=None,
                bias_orbitals=bias_orbitals,
                use_last_layer=use_last_layer,
                hf_solution=hf_solution,
                full_det=full_det,
                hidden_dims=hidden_dims,
                determinants=ndet,
                do_aa=do_aa,
                mes=mes,
                det_mode=det_mode,
                gemi_params=None,
                equal_footing=True,
                gq_type="so3",
                jastrow=jastrow,
                RHF=RHF,
                so3_config=so3_config,
            )
        )

        key, subkey = jax.random.split(key)
        params = network_init(subkey)
        return params, signed_network, det_fn, orbitals, hz


    logging.info("wfn type: %s", gq_type)

    if gq_type == "ef":
        ef = True
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
        ef = True
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
        ef = True
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
        ef = True
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

    elif gq_type == "ef_shrd_sym_proj":
        ef = True
        ferminet_model = networks.make_fermi_net_model_ef_shrd_sym_proj(
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

    elif gq_type == "ef_shrd_sym_ablation":
        ef = True
        ferminet_model = networks.make_fermi_net_model_ef_shrd_sym_ablation(
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

    elif gq_type == "ef_shrd_sym_old":
        ef = True
        ferminet_model = networks.make_fermi_net_model_ef_shrd_sym_old(
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
        ef = False
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
        ef = False
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

    else:
        raise ValueError(f"Unknown gq_type: {gq_type!r}")

    # Original make_fermi_net path stays untouched for all old network types.
    network_init, signed_network, det_fn, network_options, orbitals, hz = (
        networks.make_fermi_net(
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
    )

    key, subkey = jax.random.split(key)
    params = network_init(subkey)

    return params, signed_network, det_fn, orbitals, hz


# ----------------------------------------------------------------------
# Example: SO(3)
# ----------------------------------------------------------------------
# 训练和推断时，SO(3) 的精度由顶层 config.dtype 决定。
# 单独调用 build_network 时默认使用 float32；如需 float64，
# 须先启用 JAX x64，再传入 dtype="float64"。
#
# so3_params = {
#     "lmax": 2,
#     "channels": 16,
#     "n_layers": 3,
#     "n_rbf": 8,
#     "cutoff": None,
#     "wbn_layers": 1,
#     "wbn_bottleneck": None,
#     "target_ls": (0, 1, 2),  # 必须显式包含 l=0；也可只写 (0,)
#     "basis": "real",       # real、complex、complex_to_real
#     "init_l1": True,
#     "eps": 1.0e-12,
# }
#
# params, network, det_fn, orbitals, hz = build_network(
#     n=n,
#     charges=charges,
#     nspins=nspins,
#     key=jax.random.PRNGKey(42),
#     ndet=4,                 # K determinants FOR EACH l sector
#     depth=4,                # legacy argument; ignored by SO(3) backbone
#     h1=16,                  # legacy argument; ignored by SO(3) backbone
#     h2=16,                  # legacy argument; ignored by SO(3) backbone
#     nh=8,                   # legacy argument; ignored by SO(3) backbone
#     do_complex=False,
#     gq_type="so3",
#     so3_params=so3_params,
# )
