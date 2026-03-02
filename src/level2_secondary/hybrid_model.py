from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from merlin.algorithms.layer import CircuitBuilder, QuantumLayer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

SIM_MAX_MODES = 20
SIM_MAX_PHOTONS = 10
QPU_MAX_MODES = 24
QPU_MAX_PHOTONS = 12
QPU_SUPPORTS_AMPLITUDE_ENCODING = False
QPU_SUPPORTS_STATE_INJECTION = False


def _validate_quantum_constraints(
    q_input_dim: int,
    q_n_photons: int,
    amplitude_encoding: bool = False,
    state_injection: bool = False,
) -> Dict[str, object]:
    sim_ok = bool((q_input_dim <= SIM_MAX_MODES) and (q_n_photons <= SIM_MAX_PHOTONS))
    qpu_ok = bool(
        (q_input_dim <= QPU_MAX_MODES)
        and (q_n_photons <= QPU_MAX_PHOTONS)
        and (amplitude_encoding is False)
        and (state_injection is False)
    )

    if not sim_ok:
        raise ValueError(
            f"Simulation constraint violation: modes={q_input_dim}, photons={q_n_photons} "
            f"(limits: modes<={SIM_MAX_MODES}, photons<={SIM_MAX_PHOTONS})"
        )

    if not qpu_ok:
        raise ValueError(
            f"QPU constraint violation: modes={q_input_dim}, photons={q_n_photons}, "
            f"amplitude_encoding={amplitude_encoding}, state_injection={state_injection}. "
            f"QPU limits: modes<={QPU_MAX_MODES}, photons<={QPU_MAX_PHOTONS}, "
            "amplitude_encoding=False, state_injection=False"
        )

    return {
        "simulation": {"max_modes": SIM_MAX_MODES, "max_photons": SIM_MAX_PHOTONS, "ok": sim_ok},
        "qpu": {
            "max_modes": QPU_MAX_MODES,
            "max_photons": QPU_MAX_PHOTONS,
            "supports_amplitude_encoding": QPU_SUPPORTS_AMPLITUDE_ENCODING,
            "supports_state_injection": QPU_SUPPORTS_STATE_INJECTION,
            "ok": qpu_ok,
        },
        "selected_quantum_config": {
            "q_input_dim": int(q_input_dim),
            "q_n_photons": int(q_n_photons),
            "amplitude_encoding": bool(amplitude_encoding),
            "state_injection": bool(state_injection),
        },
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }


def make_expanding_folds(
    n_rows: int,
    n_folds: int = 5,
    min_train_frac: float = 0.50,
    val_frac: float = 0.10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    min_train = max(20, int(n_rows * min_train_frac))
    val_size = max(5, int(n_rows * val_frac))
    starts = np.linspace(min_train, n_rows - val_size, n_folds, dtype=int)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for start in starts:
        tr_idx = np.arange(0, start)
        va_idx = np.arange(start, min(n_rows, start + val_size))
        if len(tr_idx) > 0 and len(va_idx) > 1:
            folds.append((tr_idx, va_idx))

    if len(folds) != n_folds:
        raise ValueError(f"Expected {n_folds} folds, got {len(folds)}")
    return folds


def _hide(shape: Tuple[int, int], frac: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(shape) < frac


def _simplex3(step: float = 0.1) -> List[Tuple[float, float, float]]:
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    out: List[Tuple[float, float, float]] = []
    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-12:
                continue
            out.append((float(w1), float(w2), float(max(0.0, w3))))
    return out


def _project_to_qdim(x: np.ndarray, q_input_dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] == q_input_dim:
        return x
    if x.shape[1] > q_input_dim:
        return x[:, :q_input_dim]

    out = np.zeros((x.shape[0], q_input_dim), dtype=np.float64)
    out[:, : x.shape[1]] = x
    return out


def _build_merlin_quantum_layer(q_input_dim: int, q_n_photons: int, seed: int = 777) -> QuantumLayer:
    torch.manual_seed(seed)

    builder = CircuitBuilder(n_modes=q_input_dim)
    builder.add_angle_encoding(modes=list(range(q_input_dim)), scale=1.0)
    builder.add_entangling_layer(modes=None, trainable=True)

    layer = QuantumLayer(
        input_size=q_input_dim,
        builder=builder,
        n_photons=q_n_photons,
        amplitude_encoding=False,
    )
    layer.eval()
    return layer


def _merlin_quantum_features(x: np.ndarray, quantum_layer: QuantumLayer, q_input_dim: int) -> np.ndarray:
    x_proj = _project_to_qdim(np.asarray(x, dtype=np.float64), q_input_dim)
    x_t = torch.tensor(x_proj, dtype=torch.float32)
    with torch.no_grad():
        out = quantum_layer(x_t)
    return out.detach().cpu().numpy().astype(np.float64)


def _l2_feat(x_corrupt: np.ndarray, med_train: np.ndarray) -> np.ndarray:
    miss = np.isnan(x_corrupt).astype(float)
    x_fill = x_corrupt.copy()
    miss_idx = np.where(np.isnan(x_fill))
    x_fill[miss_idx] = np.take(med_train, miss_idx[1])

    row_mean = x_fill.mean(axis=1, keepdims=True)
    row_std = x_fill.std(axis=1, keepdims=True)
    row_min = x_fill.min(axis=1, keepdims=True)
    row_max = x_fill.max(axis=1, keepdims=True)
    sq = x_fill[:, : min(10, x_fill.shape[1])] ** 2

    return np.concatenate([x_fill, miss, sq, row_mean, row_std, row_min, row_max], axis=1)


def fit_l2_qhybrid(
    x_train: np.ndarray,
    seed: int = 20260228,
    q_input_dim: int = 8,
    q_n_photons: int = 3,
    min_w_q: float = 0.50,
    amplitude_encoding: bool = False,
    state_injection: bool = False,
) -> Dict[str, object]:
    x_train = np.asarray(x_train, dtype=np.float64)
    rng = np.random.default_rng(seed)

    constraint_report = _validate_quantum_constraints(
        q_input_dim=q_input_dim,
        q_n_photons=q_n_photons,
        amplitude_encoding=amplitude_encoding,
        state_injection=state_injection,
    )

    cut = max(8, int(0.80 * len(x_train)))
    x_in_tr = x_train[:cut].copy()
    x_in_va = x_train[cut:].copy()

    h_tr = _hide(x_in_tr.shape, 0.20, rng)
    h_va = _hide(x_in_va.shape, 0.20, rng)

    x_in_tr_corrupt = x_in_tr.copy()
    x_in_tr_corrupt[h_tr] = np.nan
    x_in_va_corrupt = x_in_va.copy()
    x_in_va_corrupt[h_va] = np.nan

    med = np.median(x_in_tr, axis=0)
    f_tr_raw = _l2_feat(x_in_tr_corrupt, med)
    f_va_raw = _l2_feat(x_in_va_corrupt, med)

    scaler = StandardScaler().fit(f_tr_raw)
    f_tr_s = scaler.transform(f_tr_raw)
    f_va_s = scaler.transform(f_va_raw)

    quantum_layer = _build_merlin_quantum_layer(q_input_dim=q_input_dim, q_n_photons=q_n_photons, seed=777)
    q_tr = _merlin_quantum_features(f_tr_s, quantum_layer, q_input_dim=q_input_dim)
    q_va = _merlin_quantum_features(f_va_s, quantum_layer, q_input_dim=q_input_dim)
    f_tr_q = np.hstack([f_tr_s, q_tr])
    f_va_q = np.hstack([f_va_s, q_va])

    alpha_grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    w_grid = [w for w in _simplex3(step=0.1) if w[2] >= float(min_w_q)]
    if len(w_grid) == 0:
        raise ValueError(f"No blend weights satisfy min_w_q={min_w_q}. Use a value in [0.0, 1.0].")
    lam_gap, tau_gap = 1.0, 2e-3

    best = None
    for alpha_raw in alpha_grid:
        rg_raw = Ridge(alpha=float(alpha_raw), random_state=0).fit(f_tr_s, x_in_tr)
        p_tr_raw = rg_raw.predict(f_tr_s)
        p_va_raw = rg_raw.predict(f_va_s)

        for alpha_q in alpha_grid:
            rg_q = Ridge(alpha=float(alpha_q), random_state=0).fit(f_tr_q, x_in_tr)
            p_tr_q = rg_q.predict(f_tr_q)
            p_va_q = rg_q.predict(f_va_q)

            p_tr_naive = x_in_tr_corrupt.copy()
            miss_tr = np.isnan(p_tr_naive)
            p_tr_naive[miss_tr] = np.take(med, np.where(miss_tr)[1])

            p_va_naive = x_in_va_corrupt.copy()
            miss_va = np.isnan(p_va_naive)
            p_va_naive[miss_va] = np.take(med, np.where(miss_va)[1])

            for w_naive, w_raw, w_q in w_grid:
                p_tr_h = (w_naive * p_tr_naive) + (w_raw * p_tr_raw) + (w_q * p_tr_q)
                p_va_h = (w_naive * p_va_naive) + (w_raw * p_va_raw) + (w_q * p_va_q)

                tr_rm = float(np.sqrt(np.mean((x_in_tr[h_tr] - p_tr_h[h_tr]) ** 2)))
                va_rm = float(np.sqrt(np.mean((x_in_va[h_va] - p_va_h[h_va]) ** 2)))
                obj = va_rm + lam_gap * max(0.0, (va_rm - tr_rm) - tau_gap)

                if (best is None) or (obj < best["obj"]):
                    best = {
                        "alpha_raw": float(alpha_raw),
                        "alpha_q": float(alpha_q),
                        "w_naive": float(w_naive),
                        "w_raw": float(w_raw),
                        "w_q": float(w_q),
                        "obj": float(obj),
                    }

    med_full = np.median(x_train, axis=0)
    h_full = _hide(x_train.shape, 0.20, rng)
    x_train_corrupt = x_train.copy()
    x_train_corrupt[h_full] = np.nan

    f_full_raw = _l2_feat(x_train_corrupt, med_full)
    scaler_full = StandardScaler().fit(f_full_raw)
    f_full_s = scaler_full.transform(f_full_raw)
    q_full = _merlin_quantum_features(f_full_s, quantum_layer, q_input_dim=q_input_dim)
    f_full_q = np.hstack([f_full_s, q_full])

    ridge_raw_full = Ridge(alpha=float(best["alpha_raw"]), random_state=0).fit(f_full_s, x_train)
    ridge_q_full = Ridge(alpha=float(best["alpha_q"]), random_state=0).fit(f_full_q, x_train)

    return {
        "med": med_full,
        "scaler": scaler_full,
        "ridge_raw": ridge_raw_full,
        "ridge_q": ridge_q_full,
        "alpha_raw": float(best["alpha_raw"]),
        "alpha_q": float(best["alpha_q"]),
        "w_naive": float(best["w_naive"]),
        "w_raw": float(best["w_raw"]),
        "w_q": float(best["w_q"]),
        "quantum_layer": quantum_layer,
        "q_input_dim": int(q_input_dim),
        "q_out_dim": int(q_full.shape[1]),
        "constraint_report": constraint_report,
    }


def pred_l2_qhybrid(bundle: Dict[str, object], x_in: np.ndarray) -> np.ndarray:
    x_in = np.asarray(x_in, dtype=np.float64)
    med = np.asarray(bundle["med"])

    f_raw = _l2_feat(x_in, med)
    f_s = bundle["scaler"].transform(f_raw)
    q_feat = _merlin_quantum_features(f_s, bundle["quantum_layer"], q_input_dim=int(bundle["q_input_dim"]))
    f_q = np.hstack([f_s, q_feat])

    p_raw = bundle["ridge_raw"].predict(f_s)
    p_q = bundle["ridge_q"].predict(f_q)

    p_naive = x_in.copy()
    miss = np.isnan(p_naive)
    p_naive[miss] = np.take(med, np.where(miss)[1])

    return (bundle["w_naive"] * p_naive) + (bundle["w_raw"] * p_raw) + (bundle["w_q"] * p_q)


def eval_l2_hidden(bundle: Dict[str, object], x_true: np.ndarray, frac: float = 0.20, seed: int = 1234) -> Dict[str, float]:
    x_true = np.asarray(x_true, dtype=np.float64)
    rng = np.random.default_rng(seed)
    h = _hide(x_true.shape, frac, rng)

    x_in = x_true.copy()
    x_in[h] = np.nan
    pred = pred_l2_qhybrid(bundle, x_in)
    return _metrics(x_true[h], pred[h])


def load_level2_train() -> Tuple[np.ndarray, List[str]]:
    ds = load_dataset(
        "Quandela/Challenge_Swaptions",
        data_files="level-2_Missing_data_prediction/train_level2.csv",
        split="train",
    )
    df = ds.to_pandas()
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    num_df = df.select_dtypes(include=[np.number]).copy()
    x = num_df.to_numpy(dtype=np.float64)

    if np.isnan(x).any():
        med = np.nanmedian(x, axis=0)
        miss_idx = np.where(np.isnan(x))
        x = x.copy()
        x[miss_idx] = np.take(med, miss_idx[1])

    return x, list(num_df.columns)


def fill_template(
    template_path: Path,
    out_xlsx: Path,
    out_report: Path,
    cv_repeats: int = 3,
    min_w_q: float = 0.50,
) -> Dict[str, object]:
    x_train, train_cols = load_level2_train()

    bundle = fit_l2_qhybrid(
        x_train,
        seed=20260228,
        q_input_dim=8,
        q_n_photons=3,
        min_w_q=float(min_w_q),
        amplitude_encoding=False,
        state_injection=False,
    )

    suffix = template_path.suffix.lower()
    if suffix == ".csv":
        tpl_df = pd.read_csv(template_path)
    elif suffix in {".xlsx", ".xls"}:
        tpl_df = pd.read_excel(template_path)
    else:
        raise ValueError("Unsupported template extension. Use .csv/.xlsx/.xls")

    tpl_num_df = tpl_df.select_dtypes(include=[np.number]).copy()

    missing_cols = [c for c in train_cols if c not in tpl_num_df.columns]
    extra_cols = [c for c in tpl_num_df.columns if c not in train_cols]

    x_tpl_df = tpl_num_df.copy()
    for c in missing_cols:
        x_tpl_df[c] = np.nan
    x_tpl_df = x_tpl_df[train_cols]

    x_tpl = x_tpl_df.to_numpy(dtype=np.float64)
    pred_full = pred_l2_qhybrid(bundle, x_tpl)

    true_missing_mask = np.isnan(x_tpl)
    filled_numeric = x_tpl.copy()
    filled_numeric[true_missing_mask] = pred_full[true_missing_mask]

    filled_num_df = pd.DataFrame(filled_numeric, columns=train_cols)
    out_df = tpl_df.copy()

    for col in train_cols:
        if col in out_df.columns:
            if pd.api.types.is_numeric_dtype(out_df[col]):
                miss = out_df[col].isna()
                if miss.any():
                    out_df.loc[miss, col] = filled_num_df.loc[miss, col]
        else:
            out_df[col] = filled_num_df[col]

    out_df = out_df[[c for c in tpl_df.columns] + [c for c in train_cols if c not in tpl_df.columns]]

    folds = make_expanding_folds(len(x_train), n_folds=5, min_train_frac=0.50, val_frac=0.10)
    cv_rows = []
    for rep in range(cv_repeats):
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            b = fit_l2_qhybrid(
                x_train[tr_idx],
                seed=20260228 + rep * 100 + fold_idx,
                q_input_dim=8,
                q_n_photons=3,
                min_w_q=float(min_w_q),
                amplitude_encoding=False,
                state_injection=False,
            )
            cv_rows.append(eval_l2_hidden(b, x_train[va_idx], frac=0.20, seed=4000 + rep * 100 + fold_idx))

    cv_df = pd.DataFrame(cv_rows)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_xlsx, index=False)

    report = {
        "method": "qfinger_hybrid_quantum_classical",
        "data_source": {
            "hf_dataset": "Quandela/Challenge_Swaptions",
            "data_file": "level-2_Missing_data_prediction/train_level2.csv",
            "split": "train",
        },
        "template_input": str(template_path),
        "filled_output_xlsx": str(out_xlsx),
        "rows": int(out_df.shape[0]),
        "cols": int(out_df.shape[1]),
        "train_feature_count": int(len(train_cols)),
        "true_missing_entries_filled": int(true_missing_mask.sum()),
        "missing_cols_vs_train": missing_cols,
        "extra_cols_vs_train": extra_cols,
        "hybrid_params": {
            "alpha_raw": float(bundle["alpha_raw"]),
            "alpha_q": float(bundle["alpha_q"]),
            "w_naive": float(bundle["w_naive"]),
            "w_raw": float(bundle["w_raw"]),
            "w_q": float(bundle["w_q"]),
            "min_w_q_enforced": float(min_w_q),
            "q_input_dim": int(bundle["q_input_dim"]),
            "q_n_photons": 3,
        },
        "constraints": bundle["constraint_report"],
        "repeated_cv_proxy": {
            "total_runs": int(len(cv_df)),
            "rmse_mean": float(cv_df["rmse"].mean()),
            "rmse_std": float(cv_df["rmse"].std(ddof=0)),
            "mae_mean": float(cv_df["mae"].mean()),
            "mae_std": float(cv_df["mae"].std(ddof=0)),
            "r2_mean": float(cv_df["r2"].mean()),
            "r2_std": float(cv_df["r2"].std(ddof=0)),
        },
        "references": [
            {
                "title": "Evaluating Time Series Forecasting Models",
                "authors": "Cerqueira, Torgo, Mozetic (2020)",
                "link": "https://arxiv.org/abs/1905.11744",
            },
            {
                "title": "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
                "authors": "Hoerl, Kennard (1970)",
                "link": "https://doi.org/10.1080/00401706.1970.10488634",
            },
            {
                "title": "Random Features for Large-Scale Kernel Machines",
                "authors": "Rahimi, Recht (2007)",
                "link": "https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines",
            },
            {
                "title": "Photonic Quantum-Accelerated Machine Learning",
                "authors": "arXiv:2512.08318",
                "link": "https://arxiv.org/abs/2512.08318",
            },
            {
                "title": "Unwrapping photonic reservoirs",
                "authors": "arXiv:2506.01410",
                "link": "https://arxiv.org/abs/2506.01410",
            },
            {
                "title": "Establishing Baselines for Photonic QML",
                "authors": "arXiv:2510.25839",
                "link": "https://arxiv.org/abs/2510.25839",
            },
        ],
    }

    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
