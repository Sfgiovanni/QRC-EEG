"""Crossed segment x seed bootstrap and mixed-model sensitivity analysis.

Additive follow-up to the canonical seed-averaged-then-segment gate analysis
(``scripts/make_gate_report.py``). See ``docs/crossed_inference_protocol.md``
for the frozen protocol; this module implements Sections 3-5 mechanically.
Does not modify ``scripts/make_gate_report.py`` or read/write
``results/eeg/gate_interactions.csv``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def interaction_matrix(
    df: pd.DataFrame,
    kernel: str,
    comparator: str,
    set_name: str,
    h_short: int,
    h_long: int,
    construction_col: str = "construction",
) -> tuple[np.ndarray, list, list]:
    """Return the crossed ``(segment, seed)`` interaction matrix for one
    ``(comparison, set)`` cell, without collapsing the seed axis.

    ``I[i,k] = [NRMSE_comparator(h_long,i,k) - NRMSE_comparator(h_short,i,k)]
             - [NRMSE_kernel(h_long,i,k)     - NRMSE_kernel(h_short,i,k)]``
    """

    slab = df[
        (df["set"] == set_name)
        & (df[construction_col].isin([kernel, comparator]))
        & (df["horizon"].isin([h_short, h_long]))
    ]
    pivot = slab.pivot_table(index=["segment_id", "seed"], columns=[construction_col, "horizon"], values="nrmse")
    pivot = pivot.dropna()
    segment_ids = sorted({idx[0] for idx in pivot.index})
    seeds = sorted({idx[1] for idx in pivot.index})

    kernel_d = pivot[(kernel, h_long)] - pivot[(kernel, h_short)]
    comp_d = pivot[(comparator, h_long)] - pivot[(comparator, h_short)]
    interaction = comp_d - kernel_d

    mat = interaction.unstack(level="seed")
    mat = mat.reindex(index=segment_ids, columns=seeds)
    if mat.isna().to_numpy().any():
        raise ValueError(f"incomplete segment x seed grid for {kernel} vs {comparator}, set={set_name}")
    return mat.to_numpy(dtype=float), segment_ids, seeds


def crossed_bootstrap(matrix: np.ndarray, rng: np.random.Generator, n_replicates: int = 10000) -> dict:
    """Two-factor crossed bootstrap (Section 3): independently resample the
    segment axis and the seed axis with replacement, statistic = mean over
    the Cartesian-product replica.
    """

    n_seg, n_seed = matrix.shape
    seg_idx = rng.integers(0, n_seg, size=(n_replicates, n_seg))
    seed_idx = rng.integers(0, n_seed, size=(n_replicates, n_seed))
    # sample[b] = matrix[seg_idx[b]][:, seed_idx[b]]  -- Cartesian product per replica
    sample = matrix[seg_idx[:, :, None], seed_idx[:, None, :]]  # (n_replicates, n_seg, n_seed)
    replica_means = sample.mean(axis=(1, 2))

    ci_lo, ci_hi = np.percentile(replica_means, [2.5, 97.5])
    p_le = float(np.mean(replica_means <= 0))
    p_ge = float(np.mean(replica_means >= 0))
    p_bootstrap = float(min(1.0, 2.0 * min(p_le, p_ge)))
    return {
        "observed_mean": float(matrix.mean()),
        "bootstrap_mean": float(np.mean(replica_means)),
        "bootstrap_median": float(np.median(replica_means)),
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
        "se": float(np.std(replica_means, ddof=1)),
        "sign_fraction": float(np.mean(replica_means > 0)),
        "p_bootstrap": p_bootstrap,
        "n_segments": int(n_seg),
        "n_seeds": int(n_seed),
        "n_replicates": int(n_replicates),
    }


def original_style_interaction(
    df: pd.DataFrame,
    kernel: str,
    comparator: str,
    set_name: str,
    h_short: int,
    h_long: int,
    rng: np.random.Generator,
    n_replicates: int = 10000,
    construction_col: str = "construction",
) -> dict:
    """Replicate the canonical seed-averaged-then-segment-bootstrap scheme
    (``scripts/make_gate_report.py::paired_interaction``) for side-by-side
    comparison. Uses this module's own registered RNG stream, not the
    canonical gate family's; does not read or write ``gate_interactions.csv``.
    """

    per_segment = df.groupby([construction_col, "set", "horizon", "segment_id"], as_index=False)["nrmse"].mean()
    slab = per_segment[
        (per_segment["set"] == set_name)
        & (per_segment[construction_col].isin([kernel, comparator]))
        & (per_segment["horizon"].isin([h_short, h_long]))
    ]
    pivot = slab.pivot(index="segment_id", columns=[construction_col, "horizon"], values="nrmse").dropna()
    kernel_d = pivot[(kernel, h_long)] - pivot[(kernel, h_short)]
    comp_d = pivot[(comparator, h_long)] - pivot[(comparator, h_short)]
    interaction = (comp_d - kernel_d).to_numpy(dtype=float)

    n = len(interaction)
    idx = rng.integers(0, n, size=(n_replicates, n))
    replica_means = interaction[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(replica_means, [2.5, 97.5])
    return {
        "observed_mean": float(interaction.mean()),
        "bootstrap_mean": float(np.mean(replica_means)),
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
        "n_segments": int(n),
        "n_replicates": int(n_replicates),
    }


def fit_crossed_mixed_model(
    df: pd.DataFrame,
    kernel: str,
    comparator: str,
    set_name: str,
    h_short: int,
    h_long: int,
    construction_col: str = "construction",
) -> dict:
    """Attempt a fully crossed random-intercept model

    ``nrmse ~ construction * horizon + (1|segment_id) + (1|seed)``

    via statsmodels' documented crossed-random-effects variance-components
    reformulation (no R/lme4 available in this environment). Never raises;
    returns a diagnostics dict recording convergence/singularity/boundary
    issues instead. Positive ``interaction_comp_minus_kernel`` matches the
    bootstrap sign convention (I = D_comparator - D_kernel).
    """

    import statsmodels.formula.api as smf

    out = {
        "comparison": f"{kernel} vs {comparator}", "set": set_name,
        "converged": False, "singular": None, "boundary_hit": None,
        "interaction_comp_minus_kernel": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"),
        "p_value": float("nan"), "warnings": [], "error": None,
    }

    d = df[
        (df["set"] == set_name)
        & (df[construction_col].isin([kernel, comparator]))
        & (df["horizon"].isin([h_short, h_long]))
    ].copy()
    if d.empty:
        out["error"] = "empty cell"
        return out

    d["construction_role"] = np.where(d[construction_col] == kernel, "kernel", "comparator")
    d["horizon_role"] = np.where(d["horizon"] == h_long, "long", "short")
    d["segment_id"] = d["segment_id"].astype(str)
    d["seed"] = d["seed"].astype(str)
    d["grp"] = "all"

    caught_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                "nrmse ~ C(construction_role, Treatment('comparator')) * C(horizon_role, Treatment('short'))",
                data=d,
                groups="grp",
                vc_formula={"segment": "0 + C(segment_id)", "seed": "0 + C(seed)"},
            )
            result = model.fit(reml=True)
            caught_warnings = [str(w.message) for w in wlist]

        term = "C(construction_role, Treatment('comparator'))[T.kernel]:C(horizon_role, Treatment('short'))[T.long]"
        coef = float(result.params.get(term, float("nan")))
        ci = result.conf_int()
        lo, hi = (float(ci.loc[term, 0]), float(ci.loc[term, 1])) if term in ci.index else (float("nan"), float("nan"))
        pval = float(result.pvalues.get(term, float("nan")))

        vc_var = {k: float(v) for k, v in zip(result.model.exog_vc.names, result.vcomp)} if hasattr(result, "vcomp") else {}
        warning_flags_boundary = any("boundary" in w.lower() for w in caught_warnings)
        tiny_variance = any(v < 1e-6 for v in vc_var.values()) if vc_var else False
        boundary_hit = bool(warning_flags_boundary or tiny_variance)

        cov = result.cov_params()
        cov_arr = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov)
        singular = bool(not np.all(np.isfinite(cov_arr))) or bool(np.any(np.linalg.eigvalsh(cov_arr) < 1e-12))

        # Model coefficient is D_kernel - D_comparator (kernel - comparator);
        # negate to match the bootstrap's comp_minus_kernel sign convention.
        out.update({
            "converged": bool(result.converged),
            "singular": singular,
            "boundary_hit": boundary_hit,
            "interaction_comp_minus_kernel": -coef if np.isfinite(coef) else float("nan"),
            "ci95_lo": -hi if np.isfinite(hi) else float("nan"),
            "ci95_hi": -lo if np.isfinite(lo) else float("nan"),
            "p_value": pval,
            "warnings": caught_warnings,
            "variance_components": vc_var,
        })
    except Exception as exc:  # noqa: BLE001 -- must never abort the sweep; record and move on
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["warnings"] = caught_warnings
    return out
