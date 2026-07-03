#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ethane partial-charge extrapolation probe for atom-wise nuclear hidden representation hz.

Scientific question
-------------------
For an ethane 20-configuration path, can a simple linear readout trained on early
geometries extrapolate atom-wise partial charges on later geometries?

Data expected
-------------
    hz_feature.npy : (nconfig, nwalker, natom, hidden_dim)
    atom_pos.npy   : (nconfig, natom, 3)

For ethane, natom should normally be 8 with atom order matching SYMBOLS:
    C1, C2, H1, H2, H3, H4, H5, H6

Important design choices
------------------------
1. Conventional partial charges, e.g. Lowdin or Mulliken charges, are
   configuration-level average electronic-structure quantities. They are not
   walker-level labels. Therefore the walker dimension of hz is averaged first:

       hz_mean[c, A, :] = mean_w hz[c, w, A, :]

2. The main readout is Ridge linear regression. This is intentional: if a linear
   readout extrapolates, charge information is directly/linearly readable from hz_mean.

3. The train/test split is path extrapolation, not random CV:

       train configs: 0,...,14
       test configs:  15,...,19

4. Geometry baselines are compact and physically interpretable, designed for
   extrapolation rather than high-dimensional interpolation.

Outputs
-------
    summary_ethane_charge_extrapolation.json
    ethane_lowdin_charges.npy or ethane_mulliken_charges.npy
    predictions_*.npy
    fig_charge_paths_hz_mean.png
    fig_charge_paths_compare_main.png
    fig_test_scatter_hz_mean.png
    fig_test_scatter_compact_geometry.png
    fig_test_r2_bar.png
    fig_test_mae_bar.png
    fig_per_config_mae.png
    fig_per_atom_test_mae.png
    fig_focus_<atom>_charge_path.png

Edit only the USER SETTINGS block for your machine/path/atom order.
"""

import os
import json
import math
import warnings
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# 0. USER SETTINGS FOR ETHANE
# ============================================================

# Change these paths to your ethane files.
HZ_PATH = "/Users/gaoqiao/Desktop/spring/spring_gq_1/reload_restore/hz_feature_ethane.npy"
COORDS_PATH = "/Users/gaoqiao/Desktop/spring/spring_gq_1/reload_restore/atom_pos_ethane.npy"

OUTPUT_DIR = "/Users/gaoqiao/Desktop/spring/spring_gq_1/reload_restore/hz_charge_extrapolation_ethane"

# If you already computed ethane charges, set this to the .npy path.
# The array must have shape (nconfig, natom). If None or missing, PySCF will compute it.
EXTERNAL_CHARGE_PATH = None
# Example:
# EXTERNAL_CHARGE_PATH = "/Users/gaoqiao/Desktop/spring/spring_gq_1/reload_restore_ethane/lowdin_charges.npy"

# Atom order MUST match atom_pos.npy and hz_feature.npy.
# Default ethane order: C1, C2, then six hydrogens.
SYMBOLS = ["C", "C", "H", "H", "H", "H", "H", "H"]
LABELS = ["C1", "C2", "H1", "H2", "H3", "H4", "H5", "H6"]

# Charge computation settings.
CHARGE_SCHEME = "lowdin"   # "lowdin" or "mulliken"
BASIS = "ccpvdz"
UNIT = "Bohr"              # Change to "Angstrom" if atom_pos.npy is Angstrom.
MOLECULE_CHARGE = 0
MOLECULE_SPIN = 0           # Ethane neutral closed-shell: spin = N_alpha - N_beta = 0
SCF_MAX_CYCLE = 200
SCF_CONV_TOL = 1e-10

# Path extrapolation split.
TRAIN_CONFIGS = list(range(0, 15))
TEST_CONFIGS = list(range(15, 20))

# Ridge alpha candidates. Chosen by inner blocked CV on training configs only.
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
N_INNER_FOLDS = 5

# Optional focus atoms for individual trajectory figures.
FOCUS_ATOM_LABELS = ["C1", "C2", "H1", "H4"]

EPS = 1e-8


# ============================================================
# 1. Basic utilities
# ============================================================

ELEMENT_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "S": 16,
    "Cl": 17,
}


def symbols_to_nuclear_charges(symbols: Iterable[str]) -> np.ndarray:
    try:
        return np.asarray([ELEMENT_Z[s] for s in symbols], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Element {exc} not in ELEMENT_Z. Add it to the dictionary.") from exc


def unique_in_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def validate_inputs(hz_all, coords_all, symbols, labels):
    hz_all = np.asarray(hz_all, dtype=np.float64)
    coords_all = np.asarray(coords_all, dtype=np.float64)

    if hz_all.ndim != 4:
        raise ValueError(
            f"Expected hz_all shape (nconfig, nwalker, natom, hidden_dim), got {hz_all.shape}"
        )
    if coords_all.ndim != 3:
        raise ValueError(f"Expected coords_all shape (nconfig, natom, 3), got {coords_all.shape}")

    nconfig, nwalker, natom, hidden_dim = hz_all.shape
    if coords_all.shape != (nconfig, natom, 3):
        raise ValueError(
            f"hz_all implies (nconfig, natom)=({nconfig}, {natom}), "
            f"but coords_all shape is {coords_all.shape}"
        )
    if len(symbols) != natom:
        raise ValueError(
            f"len(SYMBOLS)={len(symbols)} but natom={natom}. "
            f"Edit SYMBOLS so it matches atom_pos.npy/hz_feature.npy order."
        )
    if len(labels) != natom:
        raise ValueError(f"len(LABELS)={len(labels)} but natom={natom}")

    return hz_all, coords_all


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2 or np.var(y_true) < 1e-14:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return {
        "r2": safe_r2(y_true, y_pred),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val):
            return None
        return val
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ============================================================
# 2. PySCF partial charges
# ============================================================


def _total_density_matrix(mf):
    dm = mf.make_rdm1()
    if isinstance(dm, (tuple, list)):
        return dm[0] + dm[1]
    if getattr(dm, "ndim", None) == 3:
        return dm[0] + dm[1]
    return dm


def compute_lowdin_charge_for_mol(mol, mf) -> np.ndarray:
    dm_total = _total_density_matrix(mf)
    S = mol.intor_symmetric("int1e_ovlp")

    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-14)
    S_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    dm_lowdin = S_half @ dm_total @ S_half
    ao_pop = np.diag(dm_lowdin)

    natom = mol.natm
    aoslices = mol.aoslice_by_atom()
    atom_pop = np.zeros(natom, dtype=np.float64)
    for ia in range(natom):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        atom_pop[ia] = ao_pop[p0:p1].sum()

    nuclear_charges = np.asarray([mol.atom_charge(ia) for ia in range(natom)], dtype=np.float64)
    return nuclear_charges - atom_pop


def compute_mulliken_charge_for_mol(mol, mf) -> np.ndarray:
    dm_total = _total_density_matrix(mf)
    S = mol.intor_symmetric("int1e_ovlp")

    # Mulliken AO population: diagonal of D S in AO basis.
    ao_pop = np.einsum("ij,ji->i", dm_total, S)

    natom = mol.natm
    aoslices = mol.aoslice_by_atom()
    atom_pop = np.zeros(natom, dtype=np.float64)
    for ia in range(natom):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        atom_pop[ia] = ao_pop[p0:p1].sum()

    nuclear_charges = np.asarray([mol.atom_charge(ia) for ia in range(natom)], dtype=np.float64)
    return nuclear_charges - atom_pop


def compute_partial_charges_pyscf(
    coords_all,
    symbols,
    charge_scheme="lowdin",
    basis="ccpvdz",
    unit="Bohr",
    charge=0,
    spin=0,
    max_cycle=200,
    conv_tol=1e-10,
):
    """
    Compute UHF partial charges for each nuclear configuration.

    coords_all: (nconfig, natom, 3)
    symbols: list[str], length natom

    return: charges_all, shape (nconfig, natom)
    """
    from pyscf import gto, scf

    coords_all = np.asarray(coords_all, dtype=np.float64)
    nconfig, natom, _ = coords_all.shape
    if len(symbols) != natom:
        raise ValueError(f"len(symbols)={len(symbols)} but natom={natom}")

    scheme = charge_scheme.lower()
    if scheme not in ("lowdin", "mulliken"):
        raise ValueError("charge_scheme must be 'lowdin' or 'mulliken'")

    charges_all = []
    for iconf in range(nconfig):
        atom = [(sym, tuple(map(float, xyz))) for sym, xyz in zip(symbols, coords_all[iconf])]

        mol = gto.Mole()
        mol.atom = atom
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.unit = unit
        mol.verbose = 0
        mol.build()

        mf = scf.UHF(mol)
        mf.max_cycle = max_cycle
        mf.conv_tol = conv_tol
        mf.kernel()
        if not mf.converged:
            print(f"[Warning] UHF did not converge for config {iconf}")

        if scheme == "lowdin":
            q = compute_lowdin_charge_for_mol(mol, mf)
        else:
            q = compute_mulliken_charge_for_mol(mol, mf)

        charges_all.append(q)
        print(
            f"config {iconf:3d} | sum charge = {q.sum(): .6f} | "
            f"charges = {np.round(q, 4)}"
        )

    return np.asarray(charges_all, dtype=np.float64)


def load_or_compute_partial_charges(coords_all, symbols, output_dir):
    """Load external charges if available; otherwise compute and cache charges."""
    if EXTERNAL_CHARGE_PATH is not None and os.path.exists(EXTERNAL_CHARGE_PATH):
        print(f"Loading external partial charges from: {EXTERNAL_CHARGE_PATH}")
        charges = np.load(EXTERNAL_CHARGE_PATH)
    else:
        cache_path = os.path.join(output_dir, f"ethane_{CHARGE_SCHEME}_charges.npy")
        if os.path.exists(cache_path):
            print(f"Loading cached partial charges from: {cache_path}")
            charges = np.load(cache_path)
        else:
            charges = compute_partial_charges_pyscf(
                coords_all=coords_all,
                symbols=symbols,
                charge_scheme=CHARGE_SCHEME,
                basis=BASIS,
                unit=UNIT,
                charge=MOLECULE_CHARGE,
                spin=MOLECULE_SPIN,
                max_cycle=SCF_MAX_CYCLE,
                conv_tol=SCF_CONV_TOL,
            )
            np.save(cache_path, charges)
            print(f"Saved partial charges to: {cache_path}")

    charges = np.asarray(charges, dtype=np.float64)
    if charges.shape != coords_all.shape[:2]:
        raise ValueError(
            f"Expected charges shape {coords_all.shape[:2]}, got {charges.shape}. "
            "Check EXTERNAL_CHARGE_PATH and atom order."
        )
    return charges


# ============================================================
# 3. Geometry features designed for path extrapolation
# ============================================================


def pairwise_distances(coords_all):
    coords_all = np.asarray(coords_all, dtype=np.float64)
    diff = coords_all[:, :, None, :] - coords_all[:, None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def all_pair_indices(natom):
    return [(i, j) for i in range(natom) for j in range(i + 1, natom)]


def atom_identity_features(nconfig, symbols, labels):
    """
    Atom identity features repeated across configurations.

    Features per atom:
        atom-index one-hot, element one-hot, Z_A, Z_A^2

    For ethane this distinguishes C1/C2 and H1...H6, while also telling the model
    which atoms are chemically C or H.
    """
    natom = len(symbols)
    nuclear_charges = symbols_to_nuclear_charges(symbols)
    unique_elements = unique_in_order(symbols)

    atom_oh = np.eye(natom, dtype=np.float64)
    elem_oh = np.zeros((natom, len(unique_elements)), dtype=np.float64)
    for ia, sym in enumerate(symbols):
        elem_oh[ia, unique_elements.index(sym)] = 1.0

    Z = nuclear_charges[:, None]
    base = np.concatenate([atom_oh, elem_oh, Z, Z ** 2], axis=-1)
    return np.broadcast_to(base[None, :, :], (nconfig, natom, base.shape[-1])).copy()


def build_local_geometry_features(coords_all, symbols, labels, eps=EPS):
    """
    Low-dimensional local geometry baseline.

    Per atom A:
        atom identity + element identity + Z_A + Z_A^2
        distances r_Aj to all atoms
        inverse distances 1/r_Aj to all atoms, self inverse set to 0

    Shape: (nconfig, natom, dim)
    """
    coords_all = np.asarray(coords_all, dtype=np.float64)
    nconfig, natom, _ = coords_all.shape
    D = pairwise_distances(coords_all)
    invD = np.where(D > eps, 1.0 / (D + eps), 0.0)

    identity = atom_identity_features(nconfig, symbols, labels)
    return np.concatenate([identity, D, invD], axis=-1)


def build_compact_global_geometry_features(coords_all, symbols, labels, eps=EPS):
    """
    Compact global geometry baseline for partial-charge extrapolation.

    Per atom A:
        - atom identity + element identity + Z_A + Z_A^2
        - local distances from A to all atoms
        - local inverse distances from A to all atoms
        - all molecular pair distances r_ij, i<j
        - all molecular pair inverse distances 1/r_ij, i<j
        - global pair-distance statistics: min, max, mean, std

    This lets the baseline see the entire nuclear geometry and the identity of
    the atom whose charge is being predicted, while keeping dimension moderate.
    """
    coords_all = np.asarray(coords_all, dtype=np.float64)
    nconfig, natom, _ = coords_all.shape

    D = pairwise_distances(coords_all)
    invD = np.where(D > eps, 1.0 / (D + eps), 0.0)

    identity = atom_identity_features(nconfig, symbols, labels)

    pairs = all_pair_indices(natom)
    pair_d = np.stack([D[:, i, j] for i, j in pairs], axis=-1)
    pair_inv = 1.0 / (pair_d + eps)
    pair_stats = np.stack(
        [pair_d.min(axis=-1), pair_d.max(axis=-1), pair_d.mean(axis=-1), pair_d.std(axis=-1)],
        axis=-1,
    )

    global_feat = np.concatenate([pair_d, pair_inv, pair_stats], axis=-1)
    global_feat = np.broadcast_to(global_feat[:, None, :], (nconfig, natom, global_feat.shape[-1]))

    return np.concatenate([identity, D, invD, global_feat], axis=-1)


def build_compact_global_geometry_interaction_features(coords_all, symbols, labels, eps=EPS):
    """
    Stronger but still interpretable geometry baseline.

    Extends compact_global_geometry with atom-specific interactions:
        atom_onehot_A * global_pair_distances
        atom_onehot_A * global_pair_inverse_distances

    This lets each atom respond differently to the same global geometry. It is
    useful as a supplementary stress-test geometry baseline.
    """
    coords_all = np.asarray(coords_all, dtype=np.float64)
    nconfig, natom, _ = coords_all.shape

    base = build_compact_global_geometry_features(coords_all, symbols, labels, eps=eps)
    D = pairwise_distances(coords_all)
    pairs = all_pair_indices(natom)
    pair_d = np.stack([D[:, i, j] for i, j in pairs], axis=-1)
    pair_inv = 1.0 / (pair_d + eps)
    global_pairs = np.concatenate([pair_d, pair_inv], axis=-1)

    atom_oh = np.eye(natom, dtype=np.float64)
    interaction_blocks = []
    for ia in range(natom):
        mask = atom_oh[:, ia][None, :, None]
        block = mask * global_pairs[:, None, :]
        interaction_blocks.append(block)
    interactions = np.concatenate(interaction_blocks, axis=-1)

    return np.concatenate([base, interactions], axis=-1)


# ============================================================
# 4. Flatten atom-wise dataset and split helpers
# ============================================================


def flatten_atomwise(X_all, y_all):
    """
    X_all: (nconfig, natom, dim)
    y_all: (nconfig, natom)

    Returns:
        X:          (nconfig*natom, dim)
        y:          (nconfig*natom,)
        config_ids: (nconfig*natom,)
        atom_ids:   (nconfig*natom,)
    """
    X_all = np.asarray(X_all, dtype=np.float64)
    y_all = np.asarray(y_all, dtype=np.float64)

    if X_all.ndim != 3:
        raise ValueError(f"Expected X_all shape (nconfig, natom, dim), got {X_all.shape}")
    nconfig, natom, _ = X_all.shape
    if y_all.shape != (nconfig, natom):
        raise ValueError(f"Expected y_all shape {(nconfig, natom)}, got {y_all.shape}")

    X = X_all.reshape(nconfig * natom, -1)
    y = y_all.reshape(nconfig * natom)
    config_ids = np.repeat(np.arange(nconfig), natom)
    atom_ids = np.tile(np.arange(natom), nconfig)
    return X, y, config_ids, atom_ids


def train_test_masks(config_ids, train_configs, test_configs):
    train_configs = np.asarray(train_configs, dtype=int)
    test_configs = np.asarray(test_configs, dtype=int)
    train_mask = np.isin(config_ids, train_configs)
    test_mask = np.isin(config_ids, test_configs)
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Empty train or test split. Check TRAIN_CONFIGS and TEST_CONFIGS.")
    if np.any(train_mask & test_mask):
        raise ValueError("Train and test masks overlap.")
    return train_mask, test_mask


def blocked_inner_splits(groups_train, n_folds=5):
    """
    Contiguous group-block inner CV over training configs.

    This is used only to choose Ridge alpha. It never touches final TEST_CONFIGS.
    """
    groups_train = np.asarray(groups_train, dtype=int)
    unique_groups = np.unique(groups_train)
    n_folds = min(n_folds, len(unique_groups))
    if n_folds < 2:
        raise ValueError("Need at least two training configs for inner CV.")

    group_blocks = np.array_split(unique_groups, n_folds)
    splits = []
    for test_groups in group_blocks:
        val_mask = np.isin(groups_train, test_groups)
        train_mask = ~val_mask
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if train_idx.size > 0 and val_idx.size > 0:
            splits.append((train_idx, val_idx))
    return splits


# ============================================================
# 5. Ridge extrapolation probe
# ============================================================


def fit_predict_ridge(X_train, y_train, X_eval, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    return model, model.predict(X_eval).reshape(-1)


def choose_alpha_by_inner_cv(X_train, y_train, groups_train, alphas, n_inner_folds=5):
    splits = blocked_inner_splits(groups_train, n_folds=n_inner_folds)
    records = []

    for alpha in alphas:
        y_true_all = []
        y_pred_all = []
        for inner_train_idx, inner_val_idx in splits:
            _, y_pred = fit_predict_ridge(
                X_train[inner_train_idx],
                y_train[inner_train_idx],
                X_train[inner_val_idx],
                alpha=alpha,
            )
            y_true_all.append(y_train[inner_val_idx].reshape(-1))
            y_pred_all.append(y_pred.reshape(-1))

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        rec = regression_metrics(y_true, y_pred)
        rec["alpha"] = float(alpha)
        records.append(rec)

    # Prefer highest R2; if equal, lower MAE.
    best = sorted(records, key=lambda d: (-(d["r2"] if d["r2"] is not None else -1e99), d["mae"]))[0]
    return float(best["alpha"]), records


def per_config_metrics(y_true, y_pred, config_ids, config_list):
    out = {}
    for c in config_list:
        mask = config_ids == c
        out[int(c)] = regression_metrics(y_true[mask], y_pred[mask])
    return out


def per_atom_metrics(y_true, y_pred, atom_ids, natom):
    out = {}
    for ia in range(natom):
        mask = atom_ids == ia
        out[int(ia)] = regression_metrics(y_true[mask], y_pred[mask])
    return out


def fit_extrapolation_probe(
    X_all,
    y_all,
    config_ids,
    atom_ids,
    train_configs,
    test_configs,
    alphas,
    n_inner_folds,
    name,
):
    X_all = np.asarray(X_all, dtype=np.float64)
    y_all = np.asarray(y_all, dtype=np.float64).reshape(-1)
    config_ids = np.asarray(config_ids, dtype=int)
    atom_ids = np.asarray(atom_ids, dtype=int)

    train_mask, test_mask = train_test_masks(config_ids, train_configs, test_configs)
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    groups_train = config_ids[train_mask]

    best_alpha, alpha_records = choose_alpha_by_inner_cv(
        X_train, y_train, groups_train, alphas=alphas, n_inner_folds=n_inner_folds
    )

    model, y_pred_all = fit_predict_ridge(X_train, y_train, X_all, alpha=best_alpha)

    train_metrics = regression_metrics(y_all[train_mask], y_pred_all[train_mask])
    test_metrics = regression_metrics(y_all[test_mask], y_pred_all[test_mask])

    natom = int(atom_ids.max()) + 1
    result = {
        "name": name,
        "dim": int(X_all.shape[1]),
        "best_alpha": float(best_alpha),
        "inner_alpha_scores": alpha_records,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "per_config_test_metrics": per_config_metrics(
            y_all[test_mask], y_pred_all[test_mask], config_ids[test_mask], test_configs
        ),
        "per_atom_test_metrics": per_atom_metrics(
            y_all[test_mask], y_pred_all[test_mask], atom_ids[test_mask], natom
        ),
        "y_true_all": y_all,
        "y_pred_all": y_pred_all,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "config_ids": config_ids,
        "atom_ids": atom_ids,
    }

    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"X dim        = {X_all.shape[1]}")
    print(f"best alpha   = {best_alpha}")
    print(f"inner CV R2  = {max(r['r2'] for r in alpha_records):.6f}")
    print(f"train R2     = {train_metrics['r2']:.6f}, MAE={train_metrics['mae']:.6f}")
    print(f"test R2      = {test_metrics['r2']:.6f}, MAE={test_metrics['mae']:.6f}")

    return result


# ============================================================
# 6. Plotting
# ============================================================


def _get_model_predictions_grid(result, nconfig, natom):
    y_pred = np.asarray(result["y_pred_all"], dtype=np.float64)
    return y_pred.reshape(nconfig, natom)


def _subplot_grid(nitems: int):
    if nitems <= 4:
        ncols = 2
    elif nitems <= 8:
        ncols = 4
    else:
        ncols = 4
    nrows = int(math.ceil(nitems / ncols))
    return nrows, ncols


def plot_charge_paths(charges, pred, labels, train_configs, title, save_path):
    charges = np.asarray(charges, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    nconfig, natom = charges.shape
    x = np.arange(nconfig)

    nrows, ncols = _subplot_grid(natom)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    boundary = max(train_configs) + 0.5
    for ia, ax in enumerate(axes[:natom]):
        ax.plot(x, charges[:, ia], marker="o", linewidth=2, label="true")
        ax.plot(x, pred[:, ia], marker="s", linestyle="--", label="pred")
        ax.axvline(boundary, linestyle=":", label="train/test split" if ia == 0 else None)
        ax.set_title(labels[ia])
        ax.set_xlabel("config index")
        ax.set_ylabel("partial charge")
        ax.grid(True)
        if ia == 0:
            ax.legend(fontsize=8)
    for ax in axes[natom:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close(fig)


def plot_charge_paths_compare(charges, pred_dict, labels, train_configs, title, save_path):
    charges = np.asarray(charges, dtype=np.float64)
    nconfig, natom = charges.shape
    x = np.arange(nconfig)

    nrows, ncols = _subplot_grid(natom)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.1 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    boundary = max(train_configs) + 0.5

    for ia, ax in enumerate(axes[:natom]):
        ax.plot(x, charges[:, ia], marker="o", linewidth=2, label="true")
        for name, pred in pred_dict.items():
            ax.plot(x, pred[:, ia], linestyle="--", marker=".", label=name)
        ax.axvline(boundary, linestyle=":", label="train/test split" if ia == 0 else None)
        ax.set_title(labels[ia])
        ax.set_xlabel("config index")
        ax.set_ylabel("partial charge")
        ax.grid(True)
        if ia == 0:
            ax.legend(fontsize=8)
    for ax in axes[natom:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close(fig)


def plot_test_scatter(result, title, save_path):
    y_true = np.asarray(result["y_true_all"])[result["test_mask"]]
    y_pred = np.asarray(result["y_pred_all"])[result["test_mask"]]
    metrics = result["test_metrics"]

    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(y_true, y_pred, s=45)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.05 * (hi - lo + EPS)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")
    plt.xlabel("true partial charge, test configs")
    plt.ylabel("predicted partial charge")
    plt.title(f"{title}\nTest R2={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close()


def plot_metric_bar(results, model_names, metric, title, save_path):
    values = [results[name]["test_metrics"][metric] for name in model_names]

    plt.figure(figsize=(max(7.0, 1.4 * len(model_names)), 4.8))
    x = np.arange(len(model_names))
    plt.bar(x, values)
    plt.xticks(x, model_names, rotation=25, ha="right")
    plt.ylabel(f"test {metric}")
    plt.title(title)
    plt.grid(True, axis="y")
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            txt = "nan"
        else:
            txt = f"{v:.3f}"
        plt.text(i, 0 if np.isnan(v) else v, txt, ha="center", va="bottom" if v >= 0 else "top")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close()


def plot_per_config_mae(results, model_names, test_configs, title, save_path):
    plt.figure(figsize=(7.5, 4.5))
    for name in model_names:
        maes = [results[name]["per_config_test_metrics"][int(c)]["mae"] for c in test_configs]
        plt.plot(test_configs, maes, marker="o", label=name)
    plt.xlabel("test configuration index")
    plt.ylabel("MAE over atoms")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close()


def plot_per_atom_mae(results, model_names, labels, title, save_path):
    natom = len(labels)
    x = np.arange(natom)
    width = 0.8 / len(model_names)

    plt.figure(figsize=(max(8, 0.8 * natom + 3), 4.8))
    for k, name in enumerate(model_names):
        maes = [results[name]["per_atom_test_metrics"][ia]["mae"] for ia in range(natom)]
        plt.bar(x + (k - (len(model_names) - 1) / 2) * width, maes, width=width, label=name)
    plt.xticks(x, labels)
    plt.ylabel("test MAE")
    plt.title(title)
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close()


def plot_focus_atom(charges, pred_dict, labels, train_configs, atom_label, save_path):
    if atom_label not in labels:
        warnings.warn(f"atom_label {atom_label} not in labels; skipping focus plot.")
        return
    ia = labels.index(atom_label)
    x = np.arange(charges.shape[0])
    boundary = max(train_configs) + 0.5

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(x, charges[:, ia], marker="o", linewidth=2.5, label="true")
    for name, pred in pred_dict.items():
        plt.plot(x, pred[:, ia], marker="s", linestyle="--", label=name)
    plt.axvline(boundary, linestyle=":", linewidth=2, label="train/test split")
    plt.xlabel("configuration index")
    plt.ylabel(f"{atom_label} partial charge")
    plt.title(f"{atom_label} charge extrapolation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close()


# ============================================================
# 7. Main experiment
# ============================================================


def run_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hz_all = np.load(HZ_PATH)
    coords_all = np.load(COORDS_PATH)
    hz_all, coords_all = validate_inputs(hz_all, coords_all, SYMBOLS, LABELS)

    nconfig, nwalker, natom, hidden_dim = hz_all.shape
    print("hz:", hz_all.shape)
    print("coords:", coords_all.shape)
    print("symbols:", SYMBOLS)
    print("labels:", LABELS)
    print("train configs:", TRAIN_CONFIGS)
    print("test configs:", TEST_CONFIGS)

    charges = load_or_compute_partial_charges(coords_all, SYMBOLS, OUTPUT_DIR)
    print("charges:", charges.shape)

    # Partial charges are configuration-level labels, so walker dimension is averaged.
    hz_mean = hz_all.mean(axis=1)
    hz_std = hz_all.std(axis=1)  # optional diagnostic only
    print("hz_mean:", hz_mean.shape)

    identity = atom_identity_features(nconfig, SYMBOLS, LABELS)
    local_geo = build_local_geometry_features(coords_all, SYMBOLS, LABELS)
    compact_geo = build_compact_global_geometry_features(coords_all, SYMBOLS, LABELS)
    compact_geo_interaction = build_compact_global_geometry_interaction_features(coords_all, SYMBOLS, LABELS)

    feature_sets = {
        # Main baselines and model.
        "atom_identity": identity,
        "local_geometry": local_geo,
        "compact_geometry": compact_geo,
        "hz_mean": hz_mean,
        "compact_geometry_plus_hz": np.concatenate([compact_geo, hz_mean], axis=-1),
        # Stronger geometry stress test, useful for supplement.
        "compact_geometry_interaction": compact_geo_interaction,
        "compact_interaction_plus_hz": np.concatenate([compact_geo_interaction, hz_mean], axis=-1),
        # Optional: usually not recommended as main for configuration-level charges.
        "hz_mean_std": np.concatenate([hz_mean, hz_std], axis=-1),
    }

    _, y_flat, config_ids, atom_ids = flatten_atomwise(identity, charges)

    results = {}
    predictions_grid = {}
    for name, X_all in feature_sets.items():
        X, y, cfg, atom = flatten_atomwise(X_all, charges)
        if not np.allclose(y, y_flat) or not np.all(cfg == config_ids) or not np.all(atom == atom_ids):
            raise RuntimeError(f"Flatten mismatch for feature set {name}")
        result = fit_extrapolation_probe(
            X_all=X,
            y_all=y,
            config_ids=cfg,
            atom_ids=atom,
            train_configs=TRAIN_CONFIGS,
            test_configs=TEST_CONFIGS,
            alphas=ALPHAS,
            n_inner_folds=N_INNER_FOLDS,
            name=name,
        )
        results[name] = result
        predictions_grid[name] = _get_model_predictions_grid(result, nconfig, natom)
        np.save(os.path.join(OUTPUT_DIR, f"predictions_{name}.npy"), predictions_grid[name])

    np.save(os.path.join(OUTPUT_DIR, "hz_mean.npy"), hz_mean)
    np.save(os.path.join(OUTPUT_DIR, f"ethane_{CHARGE_SCHEME}_charges.npy"), charges)

    main_models = [
        "atom_identity",
        "local_geometry",
        "compact_geometry",
        "compact_geometry_interaction",
        "hz_mean",
        "compact_geometry_plus_hz",
    ]

    comparison_models = {
        "compact geo": predictions_grid["compact_geometry"],
        "geo interaction": predictions_grid["compact_geometry_interaction"],
        "hz_mean": predictions_grid["hz_mean"],
        "geo+hz": predictions_grid["compact_geometry_plus_hz"],
    }

    plot_charge_paths(
        charges=charges,
        pred=predictions_grid["hz_mean"],
        labels=LABELS,
        train_configs=TRAIN_CONFIGS,
        title="Ethane partial-charge extrapolation from hz_mean",
        save_path=os.path.join(OUTPUT_DIR, "fig_charge_paths_hz_mean.png"),
    )

    plot_charge_paths_compare(
        charges=charges,
        pred_dict=comparison_models,
        labels=LABELS,
        train_configs=TRAIN_CONFIGS,
        title="Ethane partial-charge extrapolation: geometry vs hz_mean",
        save_path=os.path.join(OUTPUT_DIR, "fig_charge_paths_compare_main.png"),
    )

    plot_test_scatter(
        results["hz_mean"],
        title="Ethane: hz_mean extrapolates partial charge",
        save_path=os.path.join(OUTPUT_DIR, "fig_test_scatter_hz_mean.png"),
    )

    plot_test_scatter(
        results["compact_geometry"],
        title="Ethane: compact geometry baseline",
        save_path=os.path.join(OUTPUT_DIR, "fig_test_scatter_compact_geometry.png"),
    )

    plot_test_scatter(
        results["compact_geometry_interaction"],
        title="Ethane: geometry-interaction baseline",
        save_path=os.path.join(OUTPUT_DIR, "fig_test_scatter_compact_geometry_interaction.png"),
    )

    plot_metric_bar(
        results,
        main_models,
        metric="r2",
        title="Ethane extrapolation test R2 on later geometries",
        save_path=os.path.join(OUTPUT_DIR, "fig_test_r2_bar.png"),
    )

    plot_metric_bar(
        results,
        main_models,
        metric="mae",
        title="Ethane extrapolation test MAE on later geometries",
        save_path=os.path.join(OUTPUT_DIR, "fig_test_mae_bar.png"),
    )

    plot_per_config_mae(
        results,
        ["compact_geometry", "compact_geometry_interaction", "hz_mean", "compact_geometry_plus_hz"],
        TEST_CONFIGS,
        title="Ethane test MAE as extrapolation moves along the path",
        save_path=os.path.join(OUTPUT_DIR, "fig_per_config_mae.png"),
    )

    plot_per_atom_mae(
        results,
        ["compact_geometry", "compact_geometry_interaction", "hz_mean", "compact_geometry_plus_hz"],
        LABELS,
        title="Ethane per-atom test MAE on extrapolation configs",
        save_path=os.path.join(OUTPUT_DIR, "fig_per_atom_test_mae.png"),
    )

    for atom_label in FOCUS_ATOM_LABELS:
        plot_focus_atom(
            charges=charges,
            pred_dict=comparison_models,
            labels=LABELS,
            train_configs=TRAIN_CONFIGS,
            atom_label=atom_label,
            save_path=os.path.join(OUTPUT_DIR, f"fig_focus_{atom_label}_charge_path.png"),
        )

    # Compact summary without storing large arrays in JSON.
    json_results = {}
    for name, res in results.items():
        json_results[name] = {
            "dim": res["dim"],
            "best_alpha": res["best_alpha"],
            "inner_alpha_scores": res["inner_alpha_scores"],
            "train_metrics": res["train_metrics"],
            "test_metrics": res["test_metrics"],
            "per_config_test_metrics": res["per_config_test_metrics"],
            "per_atom_test_metrics": {
                LABELS[int(k)]: v for k, v in res["per_atom_test_metrics"].items()
            },
        }

    summary = {
        "molecule": "ethane",
        "hz_shape": hz_all.shape,
        "coords_shape": coords_all.shape,
        "hz_mean_shape": hz_mean.shape,
        "charge_shape": charges.shape,
        "charge_scheme": CHARGE_SCHEME,
        "basis": BASIS,
        "unit": UNIT,
        "symbols": SYMBOLS,
        "labels": LABELS,
        "train_configs": TRAIN_CONFIGS,
        "test_configs": TEST_CONFIGS,
        "feature_dims": {name: int(X.shape[-1]) for name, X in feature_sets.items()},
        "main_recommendation": {
            "main_hz_model": "hz_mean",
            "main_geometry_baseline": "compact_geometry",
            "geometry_stress_baseline": "compact_geometry_interaction",
            "reason": (
                "Partial charge is a configuration-level quantity, so hz is averaged over walkers. "
                "compact_geometry is a low-dimensional interpretable extrapolation baseline; "
                "compact_geometry_interaction is included as a stronger geometry stress test."
            ),
        },
        "results": json_results,
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary_ethane_charge_extrapolation.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    print(f"Saved summary JSON: {summary_path}")

    print("=" * 80)
    print("Ethane compact summary")
    print("=" * 80)
    for name in main_models:
        tm = results[name]["test_metrics"]
        print(
            f"{name:32s} dim={results[name]['dim']:4d} "
            f"test R2={tm['r2']: .6f}, test MAE={tm['mae']: .6f}, "
            f"alpha={results[name]['best_alpha']}"
        )
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiment()
