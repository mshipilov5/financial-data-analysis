from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis_core import AnalysisMetric


PHASE_ORDER = ["New Moon", "Waxing", "Full Moon", "Waning"]


@dataclass
class PermutationTestResult:
    observed_diff: float
    p_value: float


def _safe_float(value: float | int | np.floating | np.integer) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


def _percentile(values: np.ndarray, p: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, p))


def _bootstrap_mean(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    return values[idx].mean(axis=1)


def bootstrap_ci_mean(
    values: Iterable[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = _bootstrap_mean(arr, n_boot=n_boot, rng=rng)
    alpha = (1.0 - ci) / 2.0
    return _percentile(means, 100 * alpha), _percentile(means, 100 * (1 - alpha))


def permutation_test_diff_means(
    a: Iterable[float],
    b: Iterable[float],
    n_perm: int = 3000,
    seed: int = 42,
) -> PermutationTestResult:
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    a_arr = a_arr[~np.isnan(a_arr)]
    b_arr = b_arr[~np.isnan(b_arr)]

    if a_arr.size < 3 or b_arr.size < 3:
        return PermutationTestResult(observed_diff=float("nan"), p_value=float("nan"))

    observed = float(a_arr.mean() - b_arr.mean())

    pooled = np.concatenate([a_arr, b_arr])
    n_a = a_arr.size
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(perm[:n_a].mean() - perm[n_a:].mean())
        if abs(diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return PermutationTestResult(observed_diff=observed, p_value=p_value)


def aggregate_by_group_equal_weight(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str = "metric",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Equal-weight aggregation: each ticker contributes equally.

    Steps:
    1) mean(metric) within each (ticker, group)
    2) average those means across tickers for each group
    """

    if df.empty:
        return pd.DataFrame(columns=[group_col, "mean", "tickers"])

    per_ticker = (
        df.groupby([ticker_col, group_col], as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "ticker_mean"})
    )

    out = (
        per_ticker.groupby(group_col, as_index=False)
        .agg(mean=("ticker_mean", "mean"), tickers=(ticker_col, "nunique"))
        .sort_values(group_col)
        .reset_index(drop=True)
    )
    return out


def build_lunar_effect_summary(
    returns_df: pd.DataFrame,
    metric: AnalysisMetric,
    mode: str = "equal_weight",
) -> Dict[str, object]:
    """Return a dict suitable for passing into Jinja and JS.

    Produces:
    - per_phase table: mean, ci_low, ci_high, obs
    - best/worst phase and permutation p-value for best-worst diff (pooled)
    """

    if returns_df.empty:
        return {
            "mode": mode,
            "phases": [],
            "best_phase": "",
            "worst_phase": "",
            "best_minus_worst": float("nan"),
            "p_value": float("nan"),
        }

    frame = returns_df.dropna(subset=["phase", "metric"]).copy()
    frame["phase"] = pd.Categorical(frame["phase"].astype(str), categories=PHASE_ORDER, ordered=True)

    if mode == "equal_weight":
        agg = aggregate_by_group_equal_weight(frame, group_col="phase", metric_col="metric")
        agg = agg.rename(columns={"tickers": "observations"})
        # CIs: bootstrap across tickers means per phase
        per_ticker = (
            frame.groupby(["ticker", "phase"], as_index=False)["metric"].mean().rename(columns={"metric": "x"})
        )
        rows = []
        for phase in PHASE_ORDER:
            x = per_ticker.loc[per_ticker["phase"] == phase, "x"].astype(float).to_numpy()
            ci_low, ci_high = bootstrap_ci_mean(x)
            mean = _safe_float(np.nanmean(x))
            rows.append(
                {
                    "phase": phase,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "observations": int(np.sum(~np.isnan(x))),
                }
            )
    else:
        grouped = frame.groupby("phase")["metric"]
        rows = []
        for phase in PHASE_ORDER:
            x = grouped.get_group(phase).astype(float).to_numpy() if phase in grouped.groups else np.array([])
            ci_low, ci_high = bootstrap_ci_mean(x)
            rows.append(
                {
                    "phase": phase,
                    "mean": _safe_float(np.nanmean(x)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "observations": int(np.sum(~np.isnan(x))),
                }
            )

    # Best/worst by mean
    valid_rows = [r for r in rows if not math.isnan(_safe_float(r["mean"]))]
    best_phase = ""
    worst_phase = ""
    best_minus_worst = float("nan")
    p_value = float("nan")

    if len(valid_rows) >= 2:
        best = max(valid_rows, key=lambda r: _safe_float(r["mean"]))
        worst = min(valid_rows, key=lambda r: _safe_float(r["mean"]))
        best_phase = str(best["phase"])
        worst_phase = str(worst["phase"])
        best_minus_worst = _safe_float(best["mean"]) - _safe_float(worst["mean"])

        # Permutation test always pooled on daily observations (simple and explainable)
        a = frame.loc[frame["phase"] == best_phase, "metric"].astype(float).to_numpy()
        b = frame.loc[frame["phase"] == worst_phase, "metric"].astype(float).to_numpy()
        test = permutation_test_diff_means(a, b)
        p_value = test.p_value

    return {
        "mode": mode,
        "metric": metric.value,
        "phases": rows,
        "best_phase": best_phase,
        "worst_phase": worst_phase,
        "best_minus_worst": best_minus_worst,
        "p_value": p_value,
    }
