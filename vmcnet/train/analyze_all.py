"""
Final article-style hz probe figures.

This script generates the figures for a two-layer story:

1. Geometry / bonding information:
   Pair-wise hz features are used to predict independently computed PySCF
   UHF/cc-pVDZ Mayer bond orders.

2. Electronic information beyond nuclear geometry:
   Walker-level hz features are used to predict within-geometry residual local
   electron-density fluctuations at fixed nuclei. Pure geometry baselines are
   constant across walkers and should have no predictive power on this target.

Expected files
--------------
hz_feature.npy : shape (nconfig, nwalker, natom, hidden_dim)
atom_pos.npy   : shape (nconfig, natom, 3)
elec_pos.npy   : shape (nconfig, nwalker, nelec, 3)

Default atom order
------------------
[C, N, O, H1, H2, H3]

Main output figures
-------------------
Figure 2 style: Mayer bond-order probe
  - fig2_mayer_selected_paths_pair_hz_only.png
  - fig2_mayer_scatter_pair_hz_only.png
  - fig2_mayer_r2_bar.png

Figure 3 style: walker-level electronic residual probe
  - fig3_rho_residual_r2_bar.png
  - fig3_rho_residual_scatter_hz_only.png
  - fig3_rho_residual_per_atom_scatter.png

Combined summary
  - fig_main_combined_summary.png

Edit the paths in the __main__ block before running.
"""

import os
import json
import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. Basic utilities
# ============================================================

ELEMENT_Z = {
    "H": 1.0,
    "C": 6.0,
    "N": 7.0,
    "O": 8.0,
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def symbols_to_z(symbols: Iterable[str]) -> np.ndarray:
    return np.asarray([ELEMENT_Z[s] for s in symbols], dtype=np.float64)


def one_hot_values(values: np.ndarray, unique_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    unique_values = np.asarray(unique_values)
    return (values[..., None] == unique_values[None, ...]).astype(np.float64)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def validate_main_inputs(
    hz_all: np.ndarray,
    atom_pos: np.ndarray,
    elec_pos: np.ndarray,
    symbols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hz_all = np.asarray(hz_all, dtype=np.float64)
    atom_pos = np.asarray(atom_pos, dtype=np.float64)
    elec_pos = np.asarray(elec_pos, dtype=np.float64)

    if hz_all.ndim != 4:
        raise ValueError(f"Expected hz_all shape (nconfig, nwalker, natom, hidden_dim), got {hz_all.shape}")
    if atom_pos.ndim != 3:
        raise ValueError(f"Expected atom_pos shape (nconfig, natom, 3), got {atom_pos.shape}")
    if elec_pos.ndim != 4:
        raise ValueError(f"Expected elec_pos shape (nconfig, nwalker, nelec, 3), got {elec_pos.shape}")

    nconfig, nwalker, natom, hidden_dim = hz_all.shape
    if atom_pos.shape != (nconfig, natom, 3):
        raise ValueError(f"Expected atom_pos shape {(nconfig, natom, 3)}, got {atom_pos.shape}")
    if elec_pos.shape[:2] != (nconfig, nwalker) or elec_pos.shape[-1] != 3:
        raise ValueError(
            f"hz_all implies (nconfig,nwalker)=({nconfig},{nwalker}), but elec_pos shape is {elec_pos.shape}"
        )
    if len(symbols) != natom:
        raise ValueError(f"len(symbols)={len(symbols)} does not match natom={natom}")

    return hz_all, atom_pos, elec_pos


# ============================================================
# 1. Generic CV probe utilities
# ============================================================


def evaluate_ridge_alpha_sweep(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    n_splits: int = 5,
    name: str = "model",
) -> Dict:
    """
    GroupKFold CV with a StandardScaler + Ridge pipeline.

    Returns the best-alpha out-of-fold prediction, metrics, and the full alpha sweep.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups)

    if X.shape[0] != y.shape[0] or X.shape[0] != groups.shape[0]:
        raise ValueError(f"Mismatched sample counts: X={X.shape}, y={y.shape}, groups={groups.shape}")

    unique_groups = np.unique(groups)
    n_splits = min(int(n_splits), len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least 2 unique groups for GroupKFold.")

    gkf = GroupKFold(n_splits=n_splits)
    split_list = list(gkf.split(X, y, groups))

    sweep = []
    best = None

    for alpha in alphas:
        y_true_all = []
        y_pred_all = []
        pred_oof = np.full_like(y, fill_value=np.nan, dtype=np.float64)

        for train_idx, test_idx in split_list:
            model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx]).reshape(-1)

            y_true_all.append(y[test_idx].reshape(-1))
            y_pred_all.append(pred)
            pred_oof[test_idx] = pred

        y_true_all = np.concatenate(y_true_all).reshape(-1)
        y_pred_all = np.concatenate(y_pred_all).reshape(-1)

        r2 = r2_score(y_true_all, y_pred_all)
        mse = mean_squared_error(y_true_all, y_pred_all)
        mae = mean_absolute_error(y_true_all, y_pred_all)

        row = {
            "alpha": float(alpha),
            "r2": float(r2),
            "mse": float(mse),
            "mae": float(mae),
        }
        sweep.append(row)

        if best is None or r2 > best["cv_r2"]:
            best = {
                "name": name,
                "dim": int(X.shape[1]),
                "best_alpha": float(alpha),
                "cv_r2": float(r2),
                "cv_mse": float(mse),
                "cv_mae": float(mae),
                "y_true": y.copy(),
                "y_pred": pred_oof,
                "alpha_sweep": sweep.copy(),
            }

    best["alpha_sweep"] = sweep

    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"X dim      = {X.shape[1]}")
    print(f"best alpha = {best['best_alpha']}")
    print(f"CV R2      = {best['cv_r2']:.6f}")
    print(f"CV MSE     = {best['cv_mse']:.6e}")
    print(f"CV MAE     = {best['cv_mae']:.6e}")

    return best


# ============================================================
# 2. Nuclear geometry features
# ============================================================


def pairwise_nuclear_distances(coords: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1) + eps)


def build_atom_geometry_simple(coords: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Return atom-wise simple geometry features, shape (nconfig, natom, 1+natom).
    Feature for atom i: [Z_i, r_i0, r_i1, ...].
    """
    coords = np.asarray(coords, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    nconfig, natom, _ = coords.shape
    D = pairwise_nuclear_distances(coords)
    Zi = np.broadcast_to(z[None, :, None], (nconfig, natom, 1))
    return np.concatenate([Zi, D], axis=-1)


def build_atom_geometry_strong(
    coords: np.ndarray,
    z: np.ndarray,
    rbf_betas: Sequence[float] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Strong rotation/translation-invariant atom-wise nuclear geometry features.

    Includes indexed distances, inverse distances, Coulomb-like terms, RBF
    expansions, element-pooled statistics, nearest-neighbor distances, and
    angular summaries. No electron coordinates are used.
    """
    coords = np.asarray(coords, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    nconfig, natom, _ = coords.shape

    D = pairwise_nuclear_distances(coords, eps=eps)
    not_self = 1.0 - np.eye(natom)[None, :, :]
    D_safe = D + (1.0 - not_self) * 1e6
    invD = not_self / (D_safe + eps)

    unique_z = np.unique(z)
    elem_oh = one_hot_values(z, unique_z)  # (A, Nelem)
    elem_oh = np.broadcast_to(elem_oh[None, :, :], (nconfig, natom, elem_oh.shape[-1]))

    Zi = np.broadcast_to(z[None, :, None], (nconfig, natom, 1))
    Zj = np.broadcast_to(z[None, None, :], (nconfig, natom, natom))

    base = [Zi, Zi ** 2, elem_oh, D, D ** 2, invD, Zj * invD, Zi * Zj * invD]

    # Indexed RBF expansion.
    rbf_indexed = [np.exp(-float(beta) * D ** 2) * not_self for beta in rbf_betas]
    base.append(np.concatenate(rbf_indexed, axis=-1))

    # Element-pooled neighbor statistics.
    pooled = []
    for zval in unique_z:
        m = ((z == zval).astype(np.float64))[None, None, :] * not_self
        count = np.maximum(m.sum(axis=-1, keepdims=True), 1.0)
        dist_mean = (D * m).sum(axis=-1, keepdims=True) / count
        dist_min = np.where(m > 0, D, 1e6).min(axis=-1, keepdims=True)
        dist_min = np.where(dist_min > 5e5, 0.0, dist_min)
        inv_sum = (invD * m).sum(axis=-1, keepdims=True)
        pooled.extend([dist_mean, dist_min, inv_sum])
        for beta in rbf_betas:
            pooled.append((np.exp(-float(beta) * D ** 2) * m).sum(axis=-1, keepdims=True))
    base.append(np.concatenate(pooled, axis=-1))

    # Sorted nearest-neighbor distances excluding self.
    D_no_self = D + (1.0 - not_self) * 1e6
    sorted_D = np.sort(D_no_self, axis=-1)
    sorted_D = sorted_D[:, :, : max(natom - 1, 1)]
    sorted_D = np.where(sorted_D > 5e5, 0.0, sorted_D)
    base.append(sorted_D)

    # Angle summaries around each center atom i.
    angle_feats = []
    for i in range(natom):
        others = [j for j in range(natom) if j != i]
        cos_list = []
        weighted_cos_list = []
        for j, k in itertools.combinations(others, 2):
            vj = coords[:, j, :] - coords[:, i, :]
            vk = coords[:, k, :] - coords[:, i, :]
            num = np.sum(vj * vk, axis=-1)
            den = np.linalg.norm(vj, axis=-1) * np.linalg.norm(vk, axis=-1) + eps
            cosang = num / den
            weight = 1.0 / ((D[:, i, j] + eps) * (D[:, i, k] + eps))
            cos_list.append(cosang[:, None])
            weighted_cos_list.append((cosang * weight)[:, None])
        if cos_list:
            coss = np.concatenate(cos_list, axis=-1)
            wcoss = np.concatenate(weighted_cos_list, axis=-1)
            feat_i = np.concatenate(
                [
                    coss.mean(axis=-1, keepdims=True),
                    coss.std(axis=-1, keepdims=True),
                    coss.min(axis=-1, keepdims=True),
                    coss.max(axis=-1, keepdims=True),
                    wcoss.mean(axis=-1, keepdims=True),
                    wcoss.std(axis=-1, keepdims=True),
                ],
                axis=-1,
            )
        else:
            feat_i = np.zeros((nconfig, 6), dtype=np.float64)
        angle_feats.append(feat_i[:, None, :])
    base.append(np.concatenate(angle_feats, axis=1))

    return np.concatenate(base, axis=-1)


# ============================================================
# 3. Mayer bond-order labels and pair features
# ============================================================


def compute_mayer_bond_orders_pyscf(
    coords_all: np.ndarray,
    symbols: Sequence[str],
    basis: str = "ccpvdz",
    charge: int = 0,
    spin: int = 0,
    unit: str = "Bohr",
    max_cycle: int = 200,
    conv_tol: float = 1e-10,
) -> np.ndarray:
    """
    Compute Mayer-style bond order matrix for each configuration using PySCF UHF.

    Formula:
      B_AB = sum_{mu in A, nu in B} (P S)_{mu nu} (P S)_{nu mu}

    P is spin-summed density matrix. Diagonal entries are set to 0.
    """
    from pyscf import gto, scf

    coords_all = np.asarray(coords_all, dtype=np.float64)
    nconfig, natom, _ = coords_all.shape
    if len(symbols) != natom:
        raise ValueError(f"len(symbols)={len(symbols)} but natom={natom}")

    mayer_all = []
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
            print(f"[Warning] UHF did not converge for Mayer BO config {iconf}")

        dm = mf.make_rdm1()
        if isinstance(dm, (tuple, list)):
            dm_total = dm[0] + dm[1]
        elif getattr(dm, "ndim", None) == 3:
            dm_total = dm[0] + dm[1]
        else:
            dm_total = dm

        S = mol.intor_symmetric("int1e_ovlp")
        PS = dm_total @ S
        aoslices = mol.aoslice_by_atom()

        bo = np.zeros((natom, natom), dtype=np.float64)
        for ia in range(natom):
            a0, a1 = aoslices[ia, 2], aoslices[ia, 3]
            idx_a = np.arange(a0, a1)
            for ib in range(ia + 1, natom):
                b0, b1 = aoslices[ib, 2], aoslices[ib, 3]
                idx_b = np.arange(b0, b1)
                block_ab = PS[np.ix_(idx_a, idx_b)]
                block_ba = PS[np.ix_(idx_b, idx_a)]
                val = float(np.sum(block_ab * block_ba.T))
                bo[ia, ib] = val
                bo[ib, ia] = val
        mayer_all.append(bo)
        print(f"Mayer config {iconf:3d}: min={bo[np.triu_indices(natom, 1)].min():.4f}, max={bo.max():.4f}")

    return np.asarray(mayer_all, dtype=np.float64)


def get_pair_indices(natom: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(natom) for j in range(i + 1, natom)]


def pair_names_from_labels(labels: Sequence[str], pair_indices: Sequence[Tuple[int, int]]) -> List[str]:
    return [f"{labels[i]}-{labels[j]}" for i, j in pair_indices]


def flatten_pair_labels(mayer_bo: np.ndarray, pair_indices: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    mayer_bo: (nconfig, natom, natom)
    return y: (nconfig*npair,), groups: (nconfig*npair,)
    """
    mayer_bo = np.asarray(mayer_bo, dtype=np.float64)
    nconfig = mayer_bo.shape[0]
    y = np.asarray([[mayer_bo[c, i, j] for (i, j) in pair_indices] for c in range(nconfig)], dtype=np.float64)
    groups = np.repeat(np.arange(nconfig), len(pair_indices))
    return y.reshape(-1), groups


def build_pair_geometry_simple(coords: np.ndarray, z: np.ndarray, pair_indices: Sequence[Tuple[int, int]]) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    nconfig, natom, _ = coords.shape
    D = pairwise_nuclear_distances(coords)
    feats = []
    for i, j in pair_indices:
        Zi = np.full((nconfig, 1), z[i])
        Zj = np.full((nconfig, 1), z[j])
        r = D[:, i, j:i + 1] if False else D[:, i, j][:, None]
        invr = 1.0 / (r + 1e-8)
        feat = np.concatenate(
            [
                Zi,
                Zj,
                Zi * Zj,
                np.abs(Zi - Zj),
                (Zi == Zj).astype(np.float64),
                r,
                r ** 2,
                invr,
            ],
            axis=-1,
        )
        feats.append(feat[:, None, :])
    return np.concatenate(feats, axis=1)  # (C, P, D)


def build_pair_from_atom_features(atom_feat: np.ndarray, pair_indices: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    atom_feat: (nconfig, natom, dim)
    pair feature: [fi, fj, |fi-fj|, fi*fj]
    return: (nconfig, npair, 4*dim)
    """
    atom_feat = np.asarray(atom_feat, dtype=np.float64)
    feats = []
    for i, j in pair_indices:
        fi = atom_feat[:, i, :]
        fj = atom_feat[:, j, :]
        feats.append(np.concatenate([fi, fj, np.abs(fi - fj), fi * fj], axis=-1)[:, None, :])
    return np.concatenate(feats, axis=1)


def flatten_pair_features(X_pair: np.ndarray) -> np.ndarray:
    X_pair = np.asarray(X_pair, dtype=np.float64)
    if X_pair.ndim != 3:
        raise ValueError(f"Expected pair feature shape (nconfig,npair,dim), got {X_pair.shape}")
    return X_pair.reshape(X_pair.shape[0] * X_pair.shape[1], X_pair.shape[2])


# ============================================================
# 4. Walker-level electronic labels and features
# ============================================================


def compute_local_density(
    atom_pos: np.ndarray,
    elec_pos: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    rho[c,w,A] = sum_e exp(-alpha * |r_e(c,w)-R_A(c)|^2)

    atom_pos: (nconfig, natom, 3)
    elec_pos: (nconfig, nwalker, nelec, 3)
    return:   (nconfig, nwalker, natom)
    """
    atom_pos = np.asarray(atom_pos, dtype=np.float64)
    elec_pos = np.asarray(elec_pos, dtype=np.float64)
    R = atom_pos[:, None, :, None, :]       # (C,1,A,1,3)
    e = elec_pos[:, :, None, :, :]          # (C,W,1,E,3)
    diff = e - R                            # (C,W,A,E,3)
    r2 = np.sum(diff * diff, axis=-1)        # (C,W,A,E)
    rho = np.sum(np.exp(-float(alpha) * r2), axis=-1)
    return rho


def within_geometry_residual(y: np.ndarray) -> np.ndarray:
    """
    y: (nconfig, nwalker, natom)
    return y - walker mean for each fixed config and atom.
    """
    y = np.asarray(y, dtype=np.float64)
    return y - y.mean(axis=1, keepdims=True)


def standardize_target(y: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float, float]:
    mu = float(np.mean(y))
    std = float(np.std(y))
    if std < eps:
        raise ValueError("Target standard deviation is too small for standardization.")
    return (y - mu) / std, mu, std


def broadcast_atom_features_to_walkers(atom_feat: np.ndarray, nwalker: int) -> np.ndarray:
    """
    atom_feat: (nconfig, natom, dim)
    return:    (nconfig, nwalker, natom, dim)
    """
    atom_feat = np.asarray(atom_feat, dtype=np.float64)
    return np.broadcast_to(atom_feat[:, None, :, :], (atom_feat.shape[0], nwalker, atom_feat.shape[1], atom_feat.shape[2]))


def flatten_walker_atom_features(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    X: (nconfig,nwalker,natom,dim)
    y: (nconfig,nwalker,natom)

    return X_flat, y_flat, groups_by_config, atom_index_flat.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 4:
        raise ValueError(f"Expected X shape (C,W,A,D), got {X.shape}")
    if y.ndim != 3:
        raise ValueError(f"Expected y shape (C,W,A), got {y.shape}")
    C, W, A, D = X.shape
    if y.shape != (C, W, A):
        raise ValueError(f"X implies y shape {(C,W,A)}, got {y.shape}")
    X_flat = X.reshape(C * W * A, D)
    y_flat = y.reshape(C * W * A)
    groups = np.repeat(np.arange(C), W * A)
    atom_idx = np.tile(np.repeat(np.arange(A), 1), C * W)
    # The above atom_idx repeats [0,1,...,A-1] for each c,w.
    atom_idx = np.tile(np.arange(A), C * W)
    return X_flat, y_flat, groups, atom_idx


# ============================================================
# 5. Plotting helpers
# ============================================================


def add_identity_line(ax, x, y):
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def savefig(path: str, dpi: int = 250) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    print(f"Saved figure: {path}")
    plt.close()


def plot_r2_bar(results: Dict[str, Dict], keys: Sequence[str], title: str, save_path: str) -> None:
    vals = [results[k]["cv_r2"] for k in keys]
    labels = [k.replace("_", "\n") for k in keys]
    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar(np.arange(len(keys)), vals)
    plt.xticks(np.arange(len(keys)), labels, rotation=0)
    plt.ylabel("Cross-validated $R^2$")
    plt.title(title)
    plt.ylim(min(-0.05, min(vals) - 0.05), min(1.05, max(vals) + 0.1))
    plt.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    savefig(save_path)


def plot_scatter_result(result: Dict, title: str, xlabel: str, ylabel: str, save_path: str, max_points: int = 20000) -> None:
    y_true = np.asarray(result["y_true"], dtype=np.float64).reshape(-1)
    y_pred = np.asarray(result["y_pred"], dtype=np.float64).reshape(-1)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y_true), size=max_points, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]

    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(y_true, y_pred, s=12, alpha=0.45)
    add_identity_line(plt.gca(), y_true, y_pred)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nCV $R^2$={result['cv_r2']:.3f}, MAE={result['cv_mae']:.3f}")
    plt.grid(alpha=0.3)
    savefig(save_path)


def plot_mayer_selected_paths(
    mayer_bo: np.ndarray,
    pred_flat: np.ndarray,
    pair_indices: Sequence[Tuple[int, int]],
    pair_names: Sequence[str],
    selected_pair_names: Sequence[str],
    save_path: str,
) -> None:
    nconfig = mayer_bo.shape[0]
    npair = len(pair_indices)
    pred = np.asarray(pred_flat, dtype=np.float64).reshape(nconfig, npair)
    true = np.asarray([[mayer_bo[c, i, j] for (i, j) in pair_indices] for c in range(nconfig)])

    name_to_idx = {name: k for k, name in enumerate(pair_names)}
    selected = [p for p in selected_pair_names if p in name_to_idx]
    if not selected:
        selected = list(pair_names[: min(6, len(pair_names))])

    ncols = 2
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.0 * nrows), squeeze=False)
    x = np.arange(nconfig)

    for ax, pname in zip(axes.ravel(), selected):
        k = name_to_idx[pname]
        ax.plot(x, true[:, k], marker="o", label="PySCF Mayer BO")
        ax.plot(x, pred[:, k], marker="s", linestyle="--", label="predicted from pair hz")
        ax.set_title(pname)
        ax.set_xlabel("configuration index")
        ax.set_ylabel("Mayer bond order")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")

    fig.suptitle("Selected Mayer bond-order trajectories: reference vs pair-hz prediction", y=1.02)
    savefig(save_path)


def plot_walker_per_atom_scatter(
    result: Dict,
    y_shape: Tuple[int, int, int],
    labels: Sequence[str],
    save_path: str,
    max_points_per_atom: int = 3000,
) -> Dict[str, Dict[str, float]]:
    C, W, A = y_shape
    y_true = np.asarray(result["y_true"], dtype=np.float64).reshape(C, W, A)
    y_pred = np.asarray(result["y_pred"], dtype=np.float64).reshape(C, W, A)

    ncols = 3
    nrows = int(np.ceil(A / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)
    rng = np.random.default_rng(0)
    metrics = {}

    for ia in range(A):
        ax = axes.ravel()[ia]
        yt = y_true[:, :, ia].reshape(-1)
        yp = y_pred[:, :, ia].reshape(-1)
        ok = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[ok], yp[ok]
        if len(yt) > max_points_per_atom:
            idx = rng.choice(len(yt), size=max_points_per_atom, replace=False)
            yt_plot, yp_plot = yt[idx], yp[idx]
        else:
            yt_plot, yp_plot = yt, yp
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        metrics[labels[ia]] = {"r2": float(r2), "mae": float(mae)}

        ax.scatter(yt_plot, yp_plot, s=8, alpha=0.35)
        add_identity_line(ax, yt_plot, yp_plot)
        ax.set_title(f"{labels[ia]}: $R^2$={r2:.3f}, MAE={mae:.3f}")
        ax.set_xlabel("true residual")
        ax.set_ylabel("predicted residual")
        ax.grid(alpha=0.3)

    for ax in axes.ravel()[A:]:
        ax.axis("off")

    fig.suptitle("Per-atom prediction of walker-level local-density residuals", y=1.02)
    savefig(save_path)
    return metrics


def plot_combined_summary(
    mayer_results: Dict[str, Dict],
    walker_results: Dict[str, Dict],
    mayer_hz_result: Dict,
    walker_hz_result: Dict,
    output_path: str,
    max_points: int = 8000,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # A: Mayer R2 bar.
    keys_m = ["pair_geometry_simple", "pair_geometry_strong", "pair_hz_only"]
    vals_m = [mayer_results[k]["cv_r2"] for k in keys_m]
    axes[0, 0].bar(np.arange(len(keys_m)), vals_m)
    axes[0, 0].set_xticks(np.arange(len(keys_m)))
    axes[0, 0].set_xticklabels(["simple\ngeo", "strong\ngeo", "pair\nhz"])
    axes[0, 0].set_ylabel("CV $R^2$")
    axes[0, 0].set_title("A. Mayer bond-order probe")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(vals_m):
        axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    # B: Mayer scatter.
    yt = np.asarray(mayer_hz_result["y_true"]).reshape(-1)
    yp = np.asarray(mayer_hz_result["y_pred"]).reshape(-1)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    axes[0, 1].scatter(yt, yp, s=18, alpha=0.5)
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    axes[0, 1].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("PySCF Mayer BO")
    axes[0, 1].set_ylabel("predicted from pair hz")
    axes[0, 1].set_title(f"B. Pair hz reconstructs BO\n$R^2$={mayer_hz_result['cv_r2']:.3f}")
    axes[0, 1].grid(alpha=0.3)

    # C: Walker residual R2 bar.
    keys_w = ["geometry_simple", "geometry_strong", "hz_only"]
    vals_w = [walker_results[k]["cv_r2"] for k in keys_w]
    axes[1, 0].bar(np.arange(len(keys_w)), vals_w)
    axes[1, 0].set_xticks(np.arange(len(keys_w)))
    axes[1, 0].set_xticklabels(["simple\ngeo", "strong\ngeo", "hz"])
    axes[1, 0].set_ylabel("CV $R^2$")
    axes[1, 0].set_title("C. Fixed-geometry electronic residual")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(vals_w):
        axes[1, 0].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)

    # D: Walker residual scatter.
    yt = np.asarray(walker_hz_result["y_true"]).reshape(-1)
    yp = np.asarray(walker_hz_result["y_pred"]).reshape(-1)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    if len(yt) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(yt), size=max_points, replace=False)
        yt, yp = yt[idx], yp[idx]
    axes[1, 1].scatter(yt, yp, s=8, alpha=0.25)
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    axes[1, 1].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("true residual local density")
    axes[1, 1].set_ylabel("predicted from hz")
    axes[1, 1].set_title(f"D. hz predicts electronic fluctuations\n$R^2$={walker_hz_result['cv_r2']:.3f}")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved figure: {output_path}")
    plt.close()


# ============================================================
# 6. Main analysis functions
# ============================================================


def run_mayer_probe(
    hz_all: np.ndarray,
    atom_pos: np.ndarray,
    symbols: Sequence[str],
    labels: Sequence[str],
    output_dir: str,
    basis: str = "ccpvdz",
    unit: str = "Bohr",
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    n_splits: int = 5,
    selected_pair_names: Sequence[str] = ("C-N", "C-O", "C-H1", "N-H1", "N-H2", "N-H3", "C-H2", "O-H2"),
) -> Tuple[Dict, np.ndarray, List[Tuple[int, int]], List[str]]:
    ensure_dir(output_dir)
    nconfig, nwalker, natom, hidden_dim = hz_all.shape
    z = symbols_to_z(symbols)
    pair_indices = get_pair_indices(natom)
    pair_names = pair_names_from_labels(labels, pair_indices)

    mayer_cache = os.path.join(output_dir, "mayer_bond_orders.npy")
    if os.path.exists(mayer_cache):
        print(f"Loading cached Mayer bond orders from: {mayer_cache}")
        mayer_bo = np.load(mayer_cache)
    else:
        mayer_bo = compute_mayer_bond_orders_pyscf(
            atom_pos,
            symbols=symbols,
            basis=basis,
            unit=unit,
            charge=0,
            spin=0,
        )
        np.save(mayer_cache, mayer_bo)
        print(f"Saved Mayer bond orders to: {mayer_cache}")

    y_pair, groups_pair = flatten_pair_labels(mayer_bo, pair_indices)

    hz_mean = hz_all.mean(axis=1)  # (C,A,H)
    atom_geo_simple = build_atom_geometry_simple(atom_pos, z)
    atom_geo_strong = build_atom_geometry_strong(atom_pos, z)

    pair_geo_simple = build_pair_geometry_simple(atom_pos, z, pair_indices)
    pair_geo_strong_from_atom = build_pair_from_atom_features(atom_geo_strong, pair_indices)
    pair_hz = build_pair_from_atom_features(hz_mean, pair_indices)

    pair_simple_plus_hz = np.concatenate([pair_geo_simple, pair_hz], axis=-1)
    pair_strong_plus_hz = np.concatenate([pair_geo_strong_from_atom, pair_hz], axis=-1)

    feature_sets = {
        "pair_geometry_simple": pair_geo_simple,
        "pair_geometry_strong": pair_geo_strong_from_atom,
        "pair_hz_only": pair_hz,
        "pair_simple_geometry_plus_hz": pair_simple_plus_hz,
        "pair_strong_geometry_plus_hz": pair_strong_plus_hz,
    }

    results = {}
    for name, Xp in feature_sets.items():
        X = flatten_pair_features(Xp)
        results[name] = evaluate_ridge_alpha_sweep(
            X,
            y_pair,
            groups_pair,
            alphas=alphas,
            n_splits=n_splits,
            name=name,
        )

    # Figures.
    plot_mayer_selected_paths(
        mayer_bo=mayer_bo,
        pred_flat=results["pair_hz_only"]["y_pred"],
        pair_indices=pair_indices,
        pair_names=pair_names,
        selected_pair_names=selected_pair_names,
        save_path=os.path.join(output_dir, "fig2_mayer_selected_paths_pair_hz_only.png"),
    )
    plot_scatter_result(
        results["pair_hz_only"],
        title="Pair-wise hz predicts Mayer bond order",
        xlabel="PySCF Mayer bond order",
        ylabel="Predicted from pair hz",
        save_path=os.path.join(output_dir, "fig2_mayer_scatter_pair_hz_only.png"),
    )
    plot_scatter_result(
        results["pair_strong_geometry_plus_hz"],
        title="Strong geometry + pair hz predicts Mayer bond order",
        xlabel="PySCF Mayer bond order",
        ylabel="Predicted from strong geometry + pair hz",
        save_path=os.path.join(output_dir, "fig2_mayer_scatter_strong_geometry_plus_hz.png"),
    )
    plot_r2_bar(
        results,
        keys=("pair_geometry_simple", "pair_geometry_strong", "pair_hz_only", "pair_strong_geometry_plus_hz"),
        title="Mayer bond-order probe",
        save_path=os.path.join(output_dir, "fig2_mayer_r2_bar.png"),
    )

    return results, mayer_bo, pair_indices, pair_names


def run_walker_electronic_probe(
    hz_all: np.ndarray,
    atom_pos: np.ndarray,
    elec_pos: np.ndarray,
    symbols: Sequence[str],
    labels: Sequence[str],
    output_dir: str,
    rho_alpha: float = 0.5,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    n_splits: int = 5,
) -> Tuple[Dict, np.ndarray]:
    ensure_dir(output_dir)
    nconfig, nwalker, natom, hidden_dim = hz_all.shape
    z = symbols_to_z(symbols)

    rho = compute_local_density(atom_pos, elec_pos, alpha=rho_alpha)  # (C,W,A)
    rho_res = within_geometry_residual(rho)
    rho_res_std, y_mu, y_std = standardize_target(rho_res)

    np.save(os.path.join(output_dir, f"rho_alpha_{rho_alpha:g}.npy"), rho)
    np.save(os.path.join(output_dir, f"rho_alpha_{rho_alpha:g}_within_geometry_residual_standardized.npy"), rho_res_std)

    atom_geo_simple = build_atom_geometry_simple(atom_pos, z)
    atom_geo_strong = build_atom_geometry_strong(atom_pos, z)

    G_simple = broadcast_atom_features_to_walkers(atom_geo_simple, nwalker)
    G_strong = broadcast_atom_features_to_walkers(atom_geo_strong, nwalker)
    H = hz_all
    GH_simple = np.concatenate([G_simple, H], axis=-1)
    GH_strong = np.concatenate([G_strong, H], axis=-1)

    feature_sets = {
        "geometry_simple": G_simple,
        "geometry_strong": G_strong,
        "hz_only": H,
        "simple_geometry_plus_hz": GH_simple,
        "strong_geometry_plus_hz": GH_strong,
    }

    results = {}
    for name, Xw in feature_sets.items():
        X, y, groups, atom_idx = flatten_walker_atom_features(Xw, rho_res_std)
        results[name] = evaluate_ridge_alpha_sweep(
            X,
            y,
            groups,
            alphas=alphas,
            n_splits=n_splits,
            name=f"rho_alpha_{rho_alpha:g}_within_geometry_residual::{name}",
        )

    # Figures.
    plot_r2_bar(
        results,
        keys=("geometry_simple", "geometry_strong", "hz_only", "strong_geometry_plus_hz"),
        title=rf"Walker-level local-density residual probe ($\alpha={rho_alpha:g}$)",
        save_path=os.path.join(output_dir, "fig3_rho_residual_r2_bar.png"),
    )
    plot_scatter_result(
        results["hz_only"],
        title=rf"hz predicts fixed-geometry local-density residuals ($\alpha={rho_alpha:g}$)",
        xlabel="True standardized residual",
        ylabel="Predicted from hz",
        save_path=os.path.join(output_dir, "fig3_rho_residual_scatter_hz_only.png"),
        max_points=25000,
    )
    per_atom_metrics = plot_walker_per_atom_scatter(
        result=results["hz_only"],
        y_shape=rho_res_std.shape,
        labels=labels,
        save_path=os.path.join(output_dir, "fig3_rho_residual_per_atom_scatter.png"),
    )
    results["hz_only"]["per_atom_metrics"] = per_atom_metrics

    return results, rho_res_std


# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Edit these paths if needed.
    # --------------------------------------------------------
    base_dir = "/Users/gaoqiao/Desktop/spring/spring_gq_1/reload_restore"

    hz_path = os.path.join(base_dir, "hz_feature.npy")
    atom_pos_path = os.path.join(base_dir, "atom_pos.npy")
    elec_pos_path = os.path.join(base_dir, "elec_pos.npy")

    output_dir = os.path.join(base_dir, "hz_final_article_figures")
    ensure_dir(output_dir)

    # Must match atom order in hz/atom_pos.
    symbols = ["C", "N", "O", "H", "H", "H"]
    labels = ["C", "N", "O", "H1", "H2", "H3"]

    # If atom_pos.npy is in Angstrom, change this to "Angstrom".
    pyscf_unit = "Bohr"
    pyscf_basis = "ccpvdz"

    # Main walker-level density residual target.
    rho_alpha = 0.5

    # Ridge regularization sweep.
    alpha_sweep = (0.1, 1.0, 10.0, 100.0, 1000.0)
    n_splits = 5

    # --------------------------------------------------------
    # Load and validate data.
    # --------------------------------------------------------
    hz_all = np.load(hz_path)
    atom_pos = np.load(atom_pos_path)
    elec_pos = np.load(elec_pos_path)

    hz_all, atom_pos, elec_pos = validate_main_inputs(hz_all, atom_pos, elec_pos, symbols)

    print("hz:", hz_all.shape)
    print("atom_pos:", atom_pos.shape)
    print("elec_pos:", elec_pos.shape)
    print("output_dir:", output_dir)

    # --------------------------------------------------------
    # Experiment 1: Mayer bond order probe.
    # --------------------------------------------------------
    mayer_dir = os.path.join(output_dir, "figure2_mayer_bond_order")
    mayer_results, mayer_bo, pair_indices, pair_names = run_mayer_probe(
        hz_all=hz_all,
        atom_pos=atom_pos,
        symbols=symbols,
        labels=labels,
        output_dir=mayer_dir,
        basis=pyscf_basis,
        unit=pyscf_unit,
        alphas=alpha_sweep,
        n_splits=n_splits,
        selected_pair_names=("C-N", "C-O", "C-H1", "N-H1", "N-H2", "N-H3", "C-H2", "O-H2"),
    )

    # --------------------------------------------------------
    # Experiment 2: walker-level local density residual probe.
    # --------------------------------------------------------
    walker_dir = os.path.join(output_dir, "figure3_walker_electronic_residual")
    walker_results, rho_res_std = run_walker_electronic_probe(
        hz_all=hz_all,
        atom_pos=atom_pos,
        elec_pos=elec_pos,
        symbols=symbols,
        labels=labels,
        output_dir=walker_dir,
        rho_alpha=rho_alpha,
        alphas=alpha_sweep,
        n_splits=n_splits,
    )

    # --------------------------------------------------------
    # Combined summary figure.
    # --------------------------------------------------------
    plot_combined_summary(
        mayer_results=mayer_results,
        walker_results=walker_results,
        mayer_hz_result=mayer_results["pair_hz_only"],
        walker_hz_result=walker_results["hz_only"],
        output_path=os.path.join(output_dir, "fig_main_combined_summary.png"),
    )

    # --------------------------------------------------------
    # Save compact summary JSON.
    # Avoid saving full y_true/y_pred arrays inside JSON; those are saved as npy.
    # --------------------------------------------------------
    compact_mayer = {}
    for k, v in mayer_results.items():
        compact_mayer[k] = {
            kk: vv for kk, vv in v.items()
            if kk not in ("y_true", "y_pred")
        }
        np.save(os.path.join(mayer_dir, f"{k}_y_true.npy"), v["y_true"])
        np.save(os.path.join(mayer_dir, f"{k}_y_pred.npy"), v["y_pred"])

    compact_walker = {}
    for k, v in walker_results.items():
        compact_walker[k] = {
            kk: vv for kk, vv in v.items()
            if kk not in ("y_true", "y_pred")
        }
        np.save(os.path.join(walker_dir, f"{k}_y_true.npy"), v["y_true"])
        np.save(os.path.join(walker_dir, f"{k}_y_pred.npy"), v["y_pred"])

    summary = {
        "hz_shape": list(hz_all.shape),
        "atom_pos_shape": list(atom_pos.shape),
        "elec_pos_shape": list(elec_pos.shape),
        "symbols": list(symbols),
        "labels": list(labels),
        "pyscf_basis": pyscf_basis,
        "pyscf_unit": pyscf_unit,
        "rho_alpha": rho_alpha,
        "pair_names": pair_names,
        "mayer_results": compact_mayer,
        "walker_rho_residual_results": compact_walker,
        "main_claim_numbers": {
            "mayer_pair_hz_only_r2": mayer_results["pair_hz_only"]["cv_r2"],
            "mayer_pair_geometry_strong_r2": mayer_results["pair_geometry_strong"]["cv_r2"],
            "walker_rho_residual_geometry_simple_r2": walker_results["geometry_simple"]["cv_r2"],
            "walker_rho_residual_geometry_strong_r2": walker_results["geometry_strong"]["cv_r2"],
            "walker_rho_residual_hz_only_r2": walker_results["hz_only"]["cv_r2"],
        },
    }

    summary_path = os.path.join(output_dir, "summary_final_article_figures.json")
    with open(summary_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    print(f"Saved summary JSON: {summary_path}")

    # --------------------------------------------------------
    # Final console summary.
    # --------------------------------------------------------
    print("=" * 80)
    print("Final article figure summary")
    print("=" * 80)
    print("Figure 2 / Mayer bond-order probe:")
    for key in ["pair_geometry_simple", "pair_geometry_strong", "pair_hz_only", "pair_strong_geometry_plus_hz"]:
        r = mayer_results[key]
        print(f"  {key:32s} R2={r['cv_r2']:.6f}, MAE={r['cv_mae']:.6e}, dim={r['dim']}")
    print("Figure 3 / Walker local-density residual probe:")
    for key in ["geometry_simple", "geometry_strong", "hz_only", "strong_geometry_plus_hz"]:
        r = walker_results[key]
        print(f"  {key:32s} R2={r['cv_r2']:.6f}, MAE={r['cv_mae']:.6e}, dim={r['dim']}")
    print(f"All outputs saved to: {output_dir}")
