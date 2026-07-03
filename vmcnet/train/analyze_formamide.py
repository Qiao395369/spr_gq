#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best-practice partial-charge extrapolation probe for atom-wise nuclear hidden representation hz.

Scientific question
-------------------
Can a simple readout trained on EARLY geometries extrapolate atom-wise partial charges
on LATER geometries along the molecular path?

Important design choices
------------------------
1. Partial charges such as Lowdin/Mulliken charges are configuration-level average
   electronic-structure quantities, not walker-level labels. Therefore hz is averaged
   over walkers first:

       hz_mean[c, A, :] = mean_w hz[c, w, A, :]

2. The main model is a linear Ridge probe. This is intentionally simple and interpretable:
   if it extrapolates, charge information is linearly readable from hz_mean.

3. The geometry baseline is redesigned for extrapolation. Instead of an overly flexible
   high-dimensional hand-crafted geometry baseline, this script uses a compact global
   geometry baseline:

       atom identity + element identity + Z_A + local atom-to-all distances
       + local inverse distances + all molecular pair distances + all inverse pair distances

   This is lower-dimensional, physically interpretable, and more stable for path extrapolation.

Flexible split
--------------
    Set TRAIN_CONFIGS / TEST_CONFIGS and TRAIN_ATOMS / TEST_ATOMS
    in the User settings section.

    Config specs can be a Python list, range(...), an integer, the string "all",
    or a string such as "0-14" or "0,1,2,7-10".

    Atom specs can be "all", atom-label strings such as "H2" or "H1,H2",
    integer atom indices such as [3, 4], or numeric ranges such as "0-5".

    Examples:
        TRAIN_CONFIGS = "0-14"
        TEST_CONFIGS  = "15-19"
        TRAIN_ATOMS   = "all"
        TEST_ATOMS    = "all"

        TRAIN_CONFIGS = [0, 10, 19]
        TEST_CONFIGS  = "all"
        TRAIN_ATOMS   = ["H2"]
        TEST_ATOMS    = ["H2"]

        TRAIN_CONFIGS = "all"
        TEST_CONFIGS  = "all"       # in-sample diagnostic
        TRAIN_ATOMS   = "H1,H2"
        TEST_ATOMS    = "H1,H2"

Default data paths
------------------
    hz_feature.npy: (20, nwalker, 6, 128)
    atom_pos.npy:   (20, 6, 3)

Outputs
-------
    summary_charge_extrapolation_best.json
    predictions_*.npy
    fig_charge_paths_*.png
    fig_test_scatter_*.png
    fig_test_r2_bar.png
    fig_test_mae_bar.png
    fig_per_config_mae.png
    fig_per_atom_test_mae.png
    fig_H2_charge_path.png
"""

import os
import json
import warnings
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# 0. User settings
# ============================================================

HZ_PATH = "/Users/gaoqiao/Desktop/spring/spring_gq_1/reload_restore/hz_feature_formamide.npy"
COORDS_PATH = "/Users/gaoqiao/Desktop/spring/spring_gq_1/reload_restore/atom_pos_formamide.npy"

OUTPUT_DIR = "/Users/gaoqiao/Desktop/spring/spring_gq_1/reload_restore/hz_charge_formamide"

# If you already computed Lowdin charges, set this path. If it does not exist,
# the script will compute charges with PySCF and save them to OUTPUT_DIR.
EXTERNAL_CHARGE_PATH = "/Users/gaoqiao/Desktop/spring/spring_gq_1/reload_restore/hz_charge_probe/lowdin_charges.npy"

SYMBOLS = ["C", "N", "O", "H", "H", "H"]
LABELS = ["C", "N", "O", "H1", "H2", "H3"]

CHARGE_SCHEME = "mulliken"   # "lowdin" or "mulliken"
BASIS = "ccpvdz"
UNIT = "Bohr"              # Change to "Angstrom" if atom_pos.npy is in Angstrom.
MOLECULE_CHARGE = 0
MOLECULE_SPIN = 0

# Flexible train/test split settings.
# You may use:
#   - list/range: list(range(0, 15)), [0, 1, 2, 5]
#   - string range: "0-14"
#   - mixed string: "0,1,2,7-10"
#   - all configs: "all"
#
# Examples:
#   TRAIN_CONFIGS = "0-14"
#   TEST_CONFIGS  = "15-19"
#
#   TRAIN_CONFIGS = [0, 2, 4, 6, 8, 10]
#   TEST_CONFIGS  = [1, 3, 5, 7, 9]
#
#   TRAIN_CONFIGS = "all"
#   TEST_CONFIGS  = "all"    # in-sample diagnostic; not an extrapolation test
TRAIN_CONFIGS = "0-10"
TEST_CONFIGS = "all"

# Flexible atom split settings.
# You may use:
#   - "all"
#   - label/list of labels: "H2", "H1,H2", ["C", "H2"]
#   - index/list of indices: 4, [3, 4, 5]
#   - numeric range string: "3-5"
#
# Examples:
#   TRAIN_ATOMS = "all"
#   TEST_ATOMS  = "all"
#
#   TRAIN_ATOMS = "H2"
#   TEST_ATOMS  = "H2"
#
#   TRAIN_ATOMS = ["H1", "H2", "H3"]
#   TEST_ATOMS  = ["H2"]
TRAIN_ATOMS = "H2"
TEST_ATOMS = "H2"

# If True, overlapping train/test samples are allowed. This is useful for diagnostics
# such as TRAIN_CONFIGS="all" and TEST_CONFIGS="all". If overlap exists, test
# metrics are no longer out-of-sample extrapolation metrics.
ALLOW_TRAIN_TEST_OVERLAP = True

# Focus-atom plots. Set to [] to skip. For ethane you might use
# FOCUS_ATOMS = ["C1", "C2", "H1", "H4"]
FOCUS_ATOMS = ["H2"]

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
N_INNER_FOLDS = 5

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
    return np.asarray([ELEMENT_Z[s] for s in symbols], dtype=np.float64)


def unique_in_order(items: Sequence[str]) -> List[str]:
    out = []
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
        raise ValueError(f"len(symbols)={len(symbols)} but natom={natom}")
    if len(labels) != natom:
        raise ValueError(f"len(labels)={len(labels)} but natom={natom}")

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


def parse_config_spec(spec, nconfig, name="CONFIGS") -> List[int]:
    """
    Parse a flexible configuration-index specification.

    Accepted forms:
      - "all"
      - int, e.g. 5
      - list/tuple/range/np.ndarray of ints, e.g. [0, 1, 2]
      - string range, e.g. "0-14"
      - comma-separated mixed string, e.g. "0,1,2,7-10"

    Returns a sorted unique list of integer config indices.
    """
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "all":
            values = list(range(nconfig))
        else:
            values = []
            for part in s.replace(" ", "").split(","):
                if part == "":
                    continue
                if "-" in part:
                    a, b = part.split("-", 1)
                    a, b = int(a), int(b)
                    step = 1 if b >= a else -1
                    values.extend(list(range(a, b + step, step)))
                else:
                    values.append(int(part))
    elif isinstance(spec, (int, np.integer)):
        values = [int(spec)]
    else:
        values = [int(x) for x in list(spec)]

    # Deduplicate while preserving input order.
    seen = set()
    out = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)

    if len(out) == 0:
        raise ValueError(f"{name} is empty after parsing: {spec!r}")

    bad = [v for v in out if v < 0 or v >= nconfig]
    if bad:
        raise ValueError(
            f"{name} contains out-of-range indices {bad}. Valid range is 0..{nconfig - 1}."
        )

    return out


def parse_atom_spec(spec, labels, name="ATOMS") -> List[int]:
    """
    Parse a flexible atom-index/atom-label specification.

    Accepted forms:
      - "all"
      - int, e.g. 4
      - label string, e.g. "H2"
      - list/tuple/range/np.ndarray containing ints and/or labels, e.g. ["H1", "H2"]
      - comma-separated mixed string, e.g. "C,H2" or "0,2,4-5"

    Returns a unique list of integer atom indices, preserving input order.
    """
    natom = len(labels)
    label_to_idx = {str(label): i for i, label in enumerate(labels)}

    def parse_one_token(token):
        token = str(token).strip()
        if token == "":
            return []
        if token in label_to_idx:
            return [label_to_idx[token]]
        low = token.lower()
        if low == "all":
            return list(range(natom))
        # Only treat a-b as a numeric range if both sides are signed integers.
        if "-" in token:
            left, right = token.split("-", 1)
            if left.lstrip("+-").isdigit() and right.lstrip("+-").isdigit():
                a, b = int(left), int(right)
                step = 1 if b >= a else -1
                return list(range(a, b + step, step))
        if token.lstrip("+-").isdigit():
            return [int(token)]
        raise ValueError(
            f"Could not parse atom token {token!r} in {name}. "
            f"Use atom labels from {labels}, integer indices, ranges like '3-5', or 'all'."
        )

    if isinstance(spec, str):
        s = spec.strip()
        if s.lower() == "all":
            values = list(range(natom))
        else:
            values = []
            for part in s.split(","):
                values.extend(parse_one_token(part))
    elif isinstance(spec, (int, np.integer)):
        values = [int(spec)]
    else:
        values = []
        for item in list(spec):
            if isinstance(item, (int, np.integer)):
                values.append(int(item))
            else:
                # Allow list entries like "H2" or "3-5".
                values.extend(parse_one_token(item))

    seen = set()
    out = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)

    if len(out) == 0:
        raise ValueError(f"{name} is empty after parsing: {spec!r}")

    bad = [v for v in out if v < 0 or v >= natom]
    if bad:
        raise ValueError(
            f"{name} contains out-of-range atom indices {bad}. "
            f"Valid range is 0..{natom - 1}; labels are {labels}."
        )

    return out


def describe_split(train_configs, test_configs, train_atoms=None, test_atoms=None, labels=None):
    train_set = set(map(int, train_configs))
    test_set = set(map(int, test_configs))
    overlap_configs = sorted(train_set & test_set)

    out = {
        "train_configs": [int(x) for x in train_configs],
        "test_configs": [int(x) for x in test_configs],
        "n_train_configs": len(train_set),
        "n_test_configs": len(test_set),
        "overlap_configs": overlap_configs,
        "has_config_overlap": len(overlap_configs) > 0,
    }

    if train_atoms is not None and test_atoms is not None:
        train_atom_set = set(map(int, train_atoms))
        test_atom_set = set(map(int, test_atoms))
        overlap_atoms = sorted(train_atom_set & test_atom_set)
        out.update({
            "train_atoms": [int(x) for x in train_atoms],
            "test_atoms": [int(x) for x in test_atoms],
            "train_atom_labels": [labels[int(x)] for x in train_atoms] if labels is not None else None,
            "test_atom_labels": [labels[int(x)] for x in test_atoms] if labels is not None else None,
            "n_train_atoms": len(train_atom_set),
            "n_test_atoms": len(test_atom_set),
            "overlap_atoms": overlap_atoms,
            "overlap_atom_labels": [labels[int(x)] for x in overlap_atoms] if labels is not None else None,
            "has_atom_overlap": len(overlap_atoms) > 0,
            "has_sample_overlap": len(overlap_configs) > 0 and len(overlap_atoms) > 0,
            "n_train_samples": len(train_set) * len(train_atom_set),
            "n_test_samples": len(test_set) * len(test_atom_set),
        })
    else:
        out["has_sample_overlap"] = out["has_config_overlap"]

    return out


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


def _lowdin_charges_from_mol_dm(mol, dm_total):
    natom = mol.natm
    S = mol.intor_symmetric("int1e_ovlp")
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-14)
    S_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    dm_lowdin = S_half @ dm_total @ S_half
    ao_pop = np.diag(dm_lowdin)

    aoslices = mol.aoslice_by_atom()
    atom_pop = np.zeros(natom, dtype=np.float64)
    for ia in range(natom):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        atom_pop[ia] = ao_pop[p0:p1].sum()

    nuclear_charges = np.asarray([mol.atom_charge(ia) for ia in range(natom)], dtype=np.float64)
    return nuclear_charges - atom_pop


def _mulliken_charges_from_mol_dm(mol, dm_total):
    natom = mol.natm
    S = mol.intor_symmetric("int1e_ovlp")
    # Mulliken gross AO population: diagonal of D S.
    ao_pop = np.diag(dm_total @ S)

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
    scheme="lowdin",
    basis="ccpvdz",
    charge=0,
    spin=0,
    unit="Bohr",
    max_cycle=200,
    conv_tol=1e-10,
):
    """
    Compute atom-wise partial charges for each nuclear geometry using PySCF UHF.

    Returns
    -------
    charges_all: np.ndarray, shape (nconfig, natom)
    """
    from pyscf import gto, scf

    coords_all = np.asarray(coords_all, dtype=np.float64)
    nconfig, natom, _ = coords_all.shape
    scheme = scheme.lower()
    if scheme not in ("lowdin", "mulliken"):
        raise ValueError(f"scheme must be 'lowdin' or 'mulliken', got {scheme}")

    charges_all = []
    for iconf in range(nconfig):
        atom = []
        for sym, xyz in zip(symbols, coords_all[iconf]):
            atom.append((sym, tuple(map(float, xyz))))

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

        dm_total = _total_density_matrix(mf)
        if scheme == "lowdin":
            q = _lowdin_charges_from_mol_dm(mol, dm_total)
        else:
            q = _mulliken_charges_from_mol_dm(mol, dm_total)

        charges_all.append(q)
        print(
            f"config {iconf:3d} | sum charge = {q.sum(): .6f} | "
            f"charges = {np.round(q, 4)}"
        )

    return np.asarray(charges_all, dtype=np.float64)


def load_or_compute_partial_charges(coords_all, symbols, output_dir):
    if EXTERNAL_CHARGE_PATH is not None and os.path.exists(EXTERNAL_CHARGE_PATH):
        print(f"Loading external partial charges from: {EXTERNAL_CHARGE_PATH}")
        charges = np.load(EXTERNAL_CHARGE_PATH)
    else:
        cache_path = os.path.join(output_dir, f"{CHARGE_SCHEME}_charges.npy")
        if os.path.exists(cache_path):
            print(f"Loading cached partial charges from: {cache_path}")
            charges = np.load(cache_path)
        else:
            print("Computing partial charges with PySCF...")
            charges = compute_partial_charges_pyscf(
                coords_all=coords_all,
                symbols=symbols,
                scheme=CHARGE_SCHEME,
                basis=BASIS,
                charge=MOLECULE_CHARGE,
                spin=MOLECULE_SPIN,
                unit=UNIT,
            )
            np.save(cache_path, charges)
            print(f"Saved partial charges to: {cache_path}")

    charges = np.asarray(charges, dtype=np.float64)
    if charges.shape != coords_all.shape[:2]:
        raise ValueError(
            f"Expected charges shape {coords_all.shape[:2]}, got {charges.shape}"
        )
    return charges


# ============================================================
# 3. Geometry features designed for extrapolation
# ============================================================


def pairwise_distances(coords_all):
    coords_all = np.asarray(coords_all, dtype=np.float64)
    diff = coords_all[:, :, None, :] - coords_all[:, None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def all_pair_indices(natom):
    return [(i, j) for i in range(natom) for j in range(i + 1, natom)]


def atom_identity_features(nconfig, symbols, labels):
    """
    Returns atom identity features repeated across configurations.

    Features per atom:
        atom-index one-hot, element one-hot, Z, Z^2
    """
    natom = len(symbols)
    nuclear_charges = symbols_to_nuclear_charges(symbols)
    unique_elements = unique_in_order(symbols)

    atom_oh = np.eye(natom, dtype=np.float64)  # distinguishes H1/H2/H3
    elem_oh = np.zeros((natom, len(unique_elements)), dtype=np.float64)
    for ia, sym in enumerate(symbols):
        elem_oh[ia, unique_elements.index(sym)] = 1.0

    Z = nuclear_charges[:, None]
    base = np.concatenate([atom_oh, elem_oh, Z, Z ** 2], axis=-1)  # (natom, dim)
    return np.broadcast_to(base[None, :, :], (nconfig, natom, base.shape[-1])).copy()


def build_local_geometry_features(coords_all, symbols, labels, eps=EPS):
    """
    A low-dimensional local geometry baseline.

    Per atom A:
        atom identity + element identity + Z_A + Z_A^2
        distances r_Aj to all atoms
        inverse distances 1/r_Aj to all atoms, with self inverse set to 0

    Shape: (nconfig, natom, dim)
    """
    coords_all = np.asarray(coords_all, dtype=np.float64)
    nconfig, natom, _ = coords_all.shape
    D = pairwise_distances(coords_all)  # (nconfig, natom, natom)
    invD = np.where(D > eps, 1.0 / (D + eps), 0.0)

    identity = atom_identity_features(nconfig, symbols, labels)
    return np.concatenate([identity, D, invD], axis=-1)


def build_compact_global_geometry_features(coords_all, symbols, labels, eps=EPS):
    """
    Compact global geometry baseline for partial-charge extrapolation.

    Per atom A, features are:
        - atom identity + element identity + Z_A + Z_A^2
        - local distances from atom A to all atoms
        - local inverse distances from atom A to all atoms
        - all molecular pair distances r_ij, i<j
        - all molecular pair inverse distances 1/r_ij, i<j
        - simple global statistics of all pair distances: min, max, mean, std

    This is compact but gives the model access to the whole nuclear geometry and
    the identity of the atom whose charge is being predicted.
    """
    coords_all = np.asarray(coords_all, dtype=np.float64)
    nconfig, natom, _ = coords_all.shape

    D = pairwise_distances(coords_all)  # (nconfig, natom, natom)
    invD = np.where(D > eps, 1.0 / (D + eps), 0.0)

    identity = atom_identity_features(nconfig, symbols, labels)

    pairs = all_pair_indices(natom)
    pair_d = np.stack([D[:, i, j] for i, j in pairs], axis=-1)  # (nconfig, npair)
    pair_inv = 1.0 / (pair_d + eps)

    pair_stats = np.stack(
        [
            pair_d.min(axis=-1),
            pair_d.max(axis=-1),
            pair_d.mean(axis=-1),
            pair_d.std(axis=-1),
        ],
        axis=-1,
    )  # (nconfig, 4)

    global_feat = np.concatenate([pair_d, pair_inv, pair_stats], axis=-1)
    global_feat = np.broadcast_to(global_feat[:, None, :], (nconfig, natom, global_feat.shape[-1]))

    return np.concatenate([identity, D, invD, global_feat], axis=-1)


def build_compact_global_geometry_interaction_features(coords_all, symbols, labels, eps=EPS):
    """
    Slightly stronger but still interpretable global geometry baseline.

    It extends compact_global_geometry with atom-specific linear interactions:
        atom_onehot_A * global_pair_distances
        atom_onehot_A * global_pair_inverse_distances

    This lets different atoms respond differently to the same global geometry
    while staying much simpler than a large RBF/angle descriptor.
    """
    coords_all = np.asarray(coords_all, dtype=np.float64)
    nconfig, natom, _ = coords_all.shape

    base = build_compact_global_geometry_features(coords_all, symbols, labels, eps=eps)
    D = pairwise_distances(coords_all)
    pairs = all_pair_indices(natom)
    pair_d = np.stack([D[:, i, j] for i, j in pairs], axis=-1)
    pair_inv = 1.0 / (pair_d + eps)
    global_pairs = np.concatenate([pair_d, pair_inv], axis=-1)  # (nconfig, 2*npair)

    atom_oh = np.eye(natom, dtype=np.float64)
    interaction_blocks = []
    for ia in range(natom):
        # For sample atom A, use the block corresponding to A; other atom blocks are zero.
        # Shape after broadcast: (nconfig, natom, 2*npair)
        mask = atom_oh[:, ia][None, :, None]
        block = mask * global_pairs[:, None, :]
        interaction_blocks.append(block)
    interactions = np.concatenate(interaction_blocks, axis=-1)

    return np.concatenate([base, interactions], axis=-1)


# ============================================================
# 4. Flatten atom-wise dataset and splits
# ============================================================


def flatten_atomwise(X_all, y_all):
    """
    X_all: (nconfig, natom, dim)
    y_all: (nconfig, natom)

    Returns X, y, config_ids, atom_ids.
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


def train_test_masks(
    config_ids,
    atom_ids,
    train_configs,
    test_configs,
    train_atoms,
    test_atoms,
    allow_overlap=ALLOW_TRAIN_TEST_OVERLAP,
):
    train_configs = np.asarray(train_configs, dtype=int)
    test_configs = np.asarray(test_configs, dtype=int)
    train_atoms = np.asarray(train_atoms, dtype=int)
    test_atoms = np.asarray(test_atoms, dtype=int)

    train_mask = np.isin(config_ids, train_configs) & np.isin(atom_ids, train_atoms)
    test_mask = np.isin(config_ids, test_configs) & np.isin(atom_ids, test_atoms)
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            "Empty train or test split after applying both config and atom selections. "
            f"train_configs={train_configs.tolist()}, test_configs={test_configs.tolist()}, "
            f"train_atoms={train_atoms.tolist()}, test_atoms={test_atoms.tolist()}"
        )

    overlap_configs = sorted(set(train_configs.tolist()) & set(test_configs.tolist()))
    overlap_atoms = sorted(set(train_atoms.tolist()) & set(test_atoms.tolist()))
    has_sample_overlap = bool(overlap_configs and overlap_atoms)

    if has_sample_overlap and not allow_overlap:
        raise ValueError(
            "Train and test samples overlap because both configs and atoms overlap. "
            f"overlap configs={overlap_configs}, overlap atoms={overlap_atoms}. "
            "Set ALLOW_TRAIN_TEST_OVERLAP=True if this is intentional."
        )
    if has_sample_overlap and allow_overlap:
        warnings.warn(
            "Train and test samples overlap because both configs and atoms overlap: "
            f"configs={overlap_configs}, atoms={overlap_atoms}. Metrics on overlapping samples "
            "are in-sample diagnostics, not strict extrapolation/generalization metrics."
        )
    elif overlap_configs and not overlap_atoms:
        warnings.warn(
            "Train and test configs overlap, but train/test atoms are disjoint. "
            "There is no identical sample overlap, but the same nuclear geometries appear in both splits."
        )

    return train_mask, test_mask


def blocked_inner_splits(groups_train, n_folds=5):
    """
    Contiguous group-block inner CV over training configs.

    This is used only to select Ridge alpha. It does not touch the final test configs.
    Returns an empty list if there are fewer than two unique training configs.
    """
    groups_train = np.asarray(groups_train, dtype=int)
    unique_groups = np.unique(groups_train)
    n_folds = min(n_folds, len(unique_groups))
    if n_folds < 2:
        return []

    group_blocks = np.array_split(unique_groups, n_folds)
    splits = []
    idx_all = np.arange(groups_train.shape[0])
    for val_groups in group_blocks:
        val_mask = np.isin(groups_train, val_groups)
        train_mask = ~val_mask
        splits.append((idx_all[train_mask], idx_all[val_mask]))
    return splits


# ============================================================
# 5. Ridge readout with inner alpha selection
# ============================================================


def fit_extrapolation_probe(
    X_all,
    y_all,
    config_ids,
    atom_ids,
    train_configs,
    test_configs,
    train_atoms,
    test_atoms,
    alphas=ALPHAS,
    n_inner_folds=N_INNER_FOLDS,
    name="model",
):
    X_all = np.asarray(X_all, dtype=np.float64)
    y_all = np.asarray(y_all, dtype=np.float64).reshape(-1)
    config_ids = np.asarray(config_ids, dtype=int)
    atom_ids = np.asarray(atom_ids, dtype=int)

    train_mask, test_mask = train_test_masks(
        config_ids=config_ids,
        atom_ids=atom_ids,
        train_configs=train_configs,
        test_configs=test_configs,
        train_atoms=train_atoms,
        test_atoms=test_atoms,
    )

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    groups_train = config_ids[train_mask]

    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    inner_splits = blocked_inner_splits(groups_train, n_folds=n_inner_folds)

    alpha_records = []
    if len(inner_splits) == 0:
        # Not enough distinct training configurations for inner CV. Use the middle alpha.
        best_alpha = float(alphas[len(alphas) // 2])
        best_rec = {"r2": float("nan"), "mse": float("nan"), "mae": float("nan"), "alpha": best_alpha}
        alpha_records.append(best_rec)
        warnings.warn(
            "Fewer than two unique training configs; inner CV for alpha selection is skipped. "
            f"Using alpha={best_alpha}."
        )
    else:
        for alpha in alphas:
            y_val_true_all = []
            y_val_pred_all = []
            for inner_train_idx, inner_val_idx in inner_splits:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(X_train[inner_train_idx], y_train[inner_train_idx])
                y_val_pred = model.predict(X_train[inner_val_idx]).reshape(-1)
                y_val_true = y_train[inner_val_idx].reshape(-1)
                y_val_true_all.append(y_val_true)
                y_val_pred_all.append(y_val_pred)

            y_val_true_all = np.concatenate(y_val_true_all)
            y_val_pred_all = np.concatenate(y_val_pred_all)
            rec = regression_metrics(y_val_true_all, y_val_pred_all)
            rec["alpha"] = float(alpha)
            alpha_records.append(rec)

        # Choose alpha by highest inner-CV R2. If R2 is nan, fall back to lowest MSE.
        valid = [r for r in alpha_records if not np.isnan(r["r2"])]
        if valid:
            best_rec = max(valid, key=lambda r: r["r2"])
        else:
            best_rec = min(alpha_records, key=lambda r: r["mse"])
        best_alpha = best_rec["alpha"]

    final_model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    final_model.fit(X_train, y_train)

    y_pred_all = final_model.predict(X_all).reshape(-1)
    y_train_pred = y_pred_all[train_mask]
    y_test_pred = y_pred_all[test_mask]

    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    per_config_test = {}
    for c in test_configs:
        mask = test_mask & (config_ids == c)
        per_config_test[int(c)] = regression_metrics(y_all[mask], y_pred_all[mask])

    per_atom_test = {}
    for ia in np.unique(atom_ids[test_mask]):
        mask = test_mask & (atom_ids == ia)
        per_atom_test[int(ia)] = regression_metrics(y_all[mask], y_pred_all[mask])

    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"dim          = {X_all.shape[1]}")
    print(f"train atoms  = {list(map(int, train_atoms))}")
    print(f"test atoms   = {list(map(int, test_atoms))}")
    print(f"best alpha   = {best_alpha}")
    print(f"inner CV R2  = {best_rec['r2']:.6f}")
    print(f"train R2     = {train_metrics['r2']:.6f}, MAE={train_metrics['mae']:.6f}")
    print(f"test R2      = {test_metrics['r2']:.6f}, MAE={test_metrics['mae']:.6f}")

    return {
        "name": name,
        "dim": int(X_all.shape[1]),
        "best_alpha": best_alpha,
        "inner_alpha_scores": alpha_records,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "per_config_test_metrics": per_config_test,
        "per_atom_test_metrics": per_atom_test,
        "y_true_all": y_all,
        "y_pred_all": y_pred_all,
        "config_ids": config_ids,
        "atom_ids": atom_ids,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }


# ============================================================
# 6. Plotting
# ============================================================


def subplot_grid(natom, max_cols=4):
    ncols = min(max_cols, natom)
    nrows = int(np.ceil(natom / ncols))
    return nrows, ncols


def annotate_split(ax, train_configs, test_configs, nconfig):
    """Add split annotation to charge-path plots.

    If train configs are all before test configs, draw one boundary line.
    Otherwise shade test configs lightly so arbitrary train/test choices are visible.
    """
    train_configs = [int(x) for x in train_configs]
    test_configs = [int(x) for x in test_configs]
    if train_configs and test_configs and max(train_configs) < min(test_configs):
        ax.axvline(max(train_configs) + 0.5, linestyle=":", label="train/test split")
    else:
        for c in test_configs:
            if 0 <= c < nconfig:
                ax.axvspan(c - 0.5, c + 0.5, alpha=0.08)


def _get_model_predictions_grid(result, nconfig, natom):
    y_pred = np.asarray(result["y_pred_all"], dtype=np.float64)
    return y_pred.reshape(nconfig, natom)


def plot_charge_paths(charges, pred, labels, train_configs, test_configs, title, save_path):
    charges = np.asarray(charges, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    nconfig, natom = charges.shape
    x = np.arange(nconfig)

    nrows, ncols = subplot_grid(natom)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for ia, ax in enumerate(axes):
        if ia >= natom:
            ax.axis("off")
            continue
        ax.plot(x, charges[:, ia], marker="o", label="true")
        ax.plot(x, pred[:, ia], marker="s", linestyle="--", label="pred")
        annotate_split(ax, train_configs, test_configs, nconfig)
        ax.set_title(labels[ia])
        ax.set_xlabel("config index")
        ax.set_ylabel("partial charge")
        ax.grid(True)
        if ia == 0:
            ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close(fig)


def plot_charge_paths_compare(charges, pred_dict, labels, train_configs, test_configs, title, save_path):
    charges = np.asarray(charges, dtype=np.float64)
    nconfig, natom = charges.shape
    x = np.arange(nconfig)

    nrows, ncols = subplot_grid(natom)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.2 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for ia, ax in enumerate(axes):
        if ia >= natom:
            ax.axis("off")
            continue
        ax.plot(x, charges[:, ia], marker="o", linewidth=2, label="true")
        for name, pred in pred_dict.items():
            ax.plot(x, pred[:, ia], linestyle="--", marker=".", label=name)
        annotate_split(ax, train_configs, test_configs, nconfig)
        ax.set_title(labels[ia])
        ax.set_xlabel("config index")
        ax.set_ylabel("partial charge")
        ax.grid(True)
        if ia == 0:
            ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close(fig)


def plot_test_scatter(result, title, save_path):
    y_true = np.asarray(result["y_true_all"])[result["test_mask"]]
    y_pred = np.asarray(result["y_pred_all"])[result["test_mask"]]
    metrics = result["test_metrics"]

    plt.figure(figsize=(5, 5))
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
    plt.figure(figsize=(8, 4.5))
    plt.bar(model_names, values)
    plt.ylabel(f"test {metric}")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close()


def plot_per_config_mae(results, model_names, test_configs, title, save_path):
    plt.figure(figsize=(7, 4.5))
    for name in model_names:
        maes = [results[name]["per_config_test_metrics"][int(c)]["mae"] for c in test_configs]
        plt.plot(test_configs, maes, marker="o", label=name)
    plt.xlabel("test config index")
    plt.ylabel("MAE over selected test atoms")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close()


def plot_per_atom_mae(results, model_names, labels, title, save_path, atom_indices=None):
    if atom_indices is None:
        atom_indices = sorted({
            int(ia)
            for name in model_names
            for ia in results[name]["per_atom_test_metrics"].keys()
        })
    atom_indices = [int(i) for i in atom_indices]
    if len(atom_indices) == 0:
        warnings.warn("No per-atom test metrics available; skipping per-atom MAE plot.")
        return

    x = np.arange(len(atom_indices))
    width = 0.8 / len(model_names)

    plt.figure(figsize=(max(7, 1.2 * len(atom_indices) + 2), 4.5))
    for k, name in enumerate(model_names):
        per_atom = results[name]["per_atom_test_metrics"]
        maes = [per_atom.get(ia, {"mae": np.nan})["mae"] for ia in atom_indices]
        plt.bar(x + (k - (len(model_names) - 1) / 2) * width, maes, width=width, label=name)
    plt.xticks(x, [labels[i] for i in atom_indices])
    plt.ylabel("test MAE")
    plt.title(title)
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    print(f"Saved figure: {save_path}")
    plt.close()


def plot_focus_atom(charges, pred_dict, labels, train_configs, test_configs, atom_label, save_path):
    if atom_label not in labels:
        warnings.warn(f"atom_label {atom_label} not in labels; skipping focus plot.")
        return
    ia = labels.index(atom_label)
    x = np.arange(charges.shape[0])

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, charges[:, ia], marker="o", linewidth=2, label="true")
    for name, pred in pred_dict.items():
        plt.plot(x, pred[:, ia], marker="s", linestyle="--", label=name)
    annotate_split(plt.gca(), train_configs, test_configs, charges.shape[0])
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
    train_configs = parse_config_spec(TRAIN_CONFIGS, nconfig, name="TRAIN_CONFIGS")
    test_configs = parse_config_spec(TEST_CONFIGS, nconfig, name="TEST_CONFIGS")
    train_atoms = parse_atom_spec(TRAIN_ATOMS, LABELS, name="TRAIN_ATOMS")
    test_atoms = parse_atom_spec(TEST_ATOMS, LABELS, name="TEST_ATOMS")
    split_info = describe_split(train_configs, test_configs, train_atoms, test_atoms, labels=LABELS)

    print("hz:", hz_all.shape)
    print("coords:", coords_all.shape)
    print("train configs:", train_configs)
    print("test configs:", test_configs)
    print("train atoms:", [LABELS[i] for i in train_atoms], train_atoms)
    print("test atoms:", [LABELS[i] for i in test_atoms], test_atoms)
    if split_info["has_sample_overlap"]:
        print(
            "[Warning] train/test samples overlap because both config and atom selections overlap. "
            "Test metrics on overlapping samples are in-sample diagnostics, not strict extrapolation metrics."
        )
    elif split_info.get("has_config_overlap"):
        print(
            "[Warning] train/test configs overlap but atom selections are disjoint. "
            "There is no identical sample overlap, but the same geometries appear in both splits."
        )

    charges = load_or_compute_partial_charges(coords_all, SYMBOLS, OUTPUT_DIR)
    print("charges:", charges.shape)

    # Partial charges are configuration-level labels, so walker dimension is averaged.
    hz_mean = hz_all.mean(axis=1)  # (nconfig, natom, hidden_dim)
    print("hz_mean:", hz_mean.shape)

    # Optional hz fluctuation statistics. This is evaluated but not recommended as main,
    # because partial charge is not walker-level.
    hz_std = hz_all.std(axis=1)

    local_geo = build_local_geometry_features(coords_all, SYMBOLS, LABELS)
    compact_geo = build_compact_global_geometry_features(coords_all, SYMBOLS, LABELS)
    compact_geo_interaction = build_compact_global_geometry_interaction_features(
        coords_all, SYMBOLS, LABELS
    )
    identity = atom_identity_features(nconfig, SYMBOLS, LABELS)

    feature_sets = {
        # Main baselines
        "atom_identity": identity,
        "local_geometry": local_geo,
        "compact_geometry": compact_geo,
        "hz_mean": hz_mean,
        "compact_geometry_plus_hz": np.concatenate([compact_geo, hz_mean], axis=-1),
        # Optional stress-test geometry baseline; useful for supplementary material.
        "compact_geometry_interaction": compact_geo_interaction,
        "compact_interaction_plus_hz": np.concatenate([compact_geo_interaction, hz_mean], axis=-1),
        # Optional: shows that adding walker std is not necessarily helpful for configuration-level charges.
        "hz_mean_std": np.concatenate([hz_mean, hz_std], axis=-1),
    }

    # Flatten once to get common ids.
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
            train_configs=train_configs,
            test_configs=test_configs,
            train_atoms=train_atoms,
            test_atoms=test_atoms,
            alphas=ALPHAS,
            n_inner_folds=N_INNER_FOLDS,
            name=name,
        )
        results[name] = result
        predictions_grid[name] = _get_model_predictions_grid(result, nconfig, natom)
        np.save(os.path.join(OUTPUT_DIR, f"predictions_{name}.npy"), predictions_grid[name])

    np.save(os.path.join(OUTPUT_DIR, "hz_mean.npy"), hz_mean)
    np.save(os.path.join(OUTPUT_DIR, f"{CHARGE_SCHEME}_charges.npy"), charges)

    # Main model set for figures.
    main_models = [
        "atom_identity",
        "local_geometry",
        "compact_geometry",
        "hz_mean",
        "compact_geometry_plus_hz",
    ]

    # Figures
    plot_charge_paths(
        charges=charges,
        pred=predictions_grid["hz_mean"],
        labels=LABELS,
        train_configs=train_configs,
        test_configs=test_configs,
        title="Partial-charge extrapolation from hz_mean",
        save_path=os.path.join(OUTPUT_DIR, "fig_charge_paths_hz_mean.png"),
    )

    plot_charge_paths_compare(
        charges=charges,
        pred_dict={
            "compact geo": predictions_grid["compact_geometry"],
            "hz_mean": predictions_grid["hz_mean"],
            "geo+hz": predictions_grid["compact_geometry_plus_hz"],
        },
        labels=LABELS,
        train_configs=train_configs,
        test_configs=test_configs,
        title="Partial-charge extrapolation: compact geometry vs hz_mean",
        save_path=os.path.join(OUTPUT_DIR, "fig_charge_paths_compare_main.png"),
    )

    plot_test_scatter(
        results["hz_mean"],
        title="hz_mean extrapolates partial charge",
        save_path=os.path.join(OUTPUT_DIR, "fig_test_scatter_hz_mean.png"),
    )

    plot_test_scatter(
        results["compact_geometry"],
        title="compact geometry baseline",
        save_path=os.path.join(OUTPUT_DIR, "fig_test_scatter_compact_geometry.png"),
    )

    plot_metric_bar(
        results,
        main_models,
        metric="r2",
        title="Extrapolation test R2 on later geometries",
        save_path=os.path.join(OUTPUT_DIR, "fig_test_r2_bar.png"),
    )

    plot_metric_bar(
        results,
        main_models,
        metric="mae",
        title="Extrapolation test MAE on later geometries",
        save_path=os.path.join(OUTPUT_DIR, "fig_test_mae_bar.png"),
    )

    plot_per_config_mae(
        results,
        ["compact_geometry", "hz_mean", "compact_geometry_plus_hz"],
        test_configs,
        title="Test MAE as extrapolation moves farther along the path",
        save_path=os.path.join(OUTPUT_DIR, "fig_per_config_mae.png"),
    )

    plot_per_atom_mae(
        results,
        ["compact_geometry", "hz_mean", "compact_geometry_plus_hz"],
        LABELS,
        title="Per-atom test MAE on selected test samples",
        save_path=os.path.join(OUTPUT_DIR, "fig_per_atom_test_mae.png"),
        atom_indices=test_atoms,
    )

    for atom_label in FOCUS_ATOMS:
        plot_focus_atom(
            charges=charges,
            pred_dict={
                "compact geo": predictions_grid["compact_geometry"],
                "hz_mean": predictions_grid["hz_mean"],
                "geo+hz": predictions_grid["compact_geometry_plus_hz"],
            },
            labels=LABELS,
            train_configs=train_configs,
            test_configs=test_configs,
            atom_label=atom_label,
            save_path=os.path.join(OUTPUT_DIR, f"fig_{atom_label}_charge_path.png"),
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
        "hz_shape": hz_all.shape,
        "coords_shape": coords_all.shape,
        "hz_mean_shape": hz_mean.shape,
        "charge_shape": charges.shape,
        "charge_scheme": CHARGE_SCHEME,
        "basis": BASIS,
        "unit": UNIT,
        "symbols": SYMBOLS,
        "labels": LABELS,
        "train_config_spec": to_jsonable(TRAIN_CONFIGS),
        "test_config_spec": to_jsonable(TEST_CONFIGS),
        "train_atom_spec": to_jsonable(TRAIN_ATOMS),
        "test_atom_spec": to_jsonable(TEST_ATOMS),
        "train_configs": train_configs,
        "test_configs": test_configs,
        "train_atoms": train_atoms,
        "test_atoms": test_atoms,
        "train_atom_labels": [LABELS[int(i)] for i in train_atoms],
        "test_atom_labels": [LABELS[int(i)] for i in test_atoms],
        "allow_train_test_overlap": ALLOW_TRAIN_TEST_OVERLAP,
        "split_info": split_info,
        "feature_dims": {name: int(X.shape[-1]) for name, X in feature_sets.items()},
        "main_recommendation": {
            "main_geometry_baseline": "compact_geometry",
            "main_hz_model": "hz_mean",
            "main_combined_model": "compact_geometry_plus_hz",
            "reason": (
                "compact_geometry is a lower-dimensional, physically interpretable path-extrapolation "
                "baseline; hz_mean is the walker-averaged representation appropriate for configuration-level "
                "partial charges."
            ),
        },
        "results": json_results,
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary_charge_extrapolation_best.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    print(f"Saved summary JSON: {summary_path}")

    print("=" * 80)
    print("Compact summary")
    print("=" * 80)
    print(f"Train configs: {train_configs}")
    print(f"Test configs : {test_configs}")
    print(f"Train atoms  : {[LABELS[i] for i in train_atoms]}")
    print(f"Test atoms   : {[LABELS[i] for i in test_atoms]}")
    for name in main_models:
        tm = results[name]["test_metrics"]
        print(
            f"{name:28s} dim={results[name]['dim']:4d} "
            f"test R2={tm['r2']: .6f}, test MAE={tm['mae']: .6f}, "
            f"alpha={results[name]['best_alpha']}"
        )
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiment()
