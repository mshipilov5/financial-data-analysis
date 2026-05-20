from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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
    except (TypeError, ValueError, OverflowError):
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


def build_halloween_effect_summary(
    returns_df: pd.DataFrame,
    halloween_by_ticker: pd.DataFrame,
    metric: AnalysisMetric,
) -> Dict[str, object]:
    if returns_df.empty:
        return {
            "winter_mean": float("nan"),
            "summer_mean": float("nan"),
            "winter_minus_summer": float("nan"),
            "p_value": float("nan"),
            "tickers_total": 0,
            "tickers_winter_better": 0,
            "share_winter_better": float("nan"),
            "by_ticker": [],
            "metric": metric.value,
        }

    frame = returns_df.dropna(subset=["date", "metric"]).copy()
    month = pd.to_datetime(frame["date"]).dt.month
    frame["season"] = np.where(
        month.isin([11, 12, 1, 2, 3, 4]),
        "Winter (Nov-Apr)",
        "Summer (May-Oct)",
    )

    winter = frame.loc[frame["season"] == "Winter (Nov-Apr)", "metric"].astype(float).to_numpy()
    summer = frame.loc[frame["season"] == "Summer (May-Oct)", "metric"].astype(float).to_numpy()
    winter_mean = _safe_float(np.nanmean(winter))
    summer_mean = _safe_float(np.nanmean(summer))
    test = permutation_test_diff_means(winter, summer)

    by_ticker_df = halloween_by_ticker.copy()
    by_ticker_df = by_ticker_df.dropna(subset=["ticker", "avg_daily_return_diff"])
    by_ticker_df = by_ticker_df.sort_values("ticker")
    tickers_total = int(len(by_ticker_df))
    tickers_winter_better = int((by_ticker_df["avg_daily_return_diff"] > 0).sum()) if tickers_total else 0
    share_winter_better = (
        _safe_float(tickers_winter_better / tickers_total) if tickers_total else float("nan")
    )

    return {
        "winter_mean": winter_mean,
        "summer_mean": summer_mean,
        "winter_minus_summer": winter_mean - summer_mean,
        "p_value": test.p_value,
        "tickers_total": tickers_total,
        "tickers_winter_better": tickers_winter_better,
        "share_winter_better": share_winter_better,
        "by_ticker": [
            {
                "ticker": str(row["ticker"]),
                "avg_daily_return_diff": _safe_float(row["avg_daily_return_diff"]),
                "total_return_diff": _safe_float(row["total_return_diff"]),
            }
            for _, row in by_ticker_df.iterrows()
        ],
        "metric": metric.value,
    }


def _build_weather_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    temp_order = ["Холодно", "Умеренно", "Тепло"]
    rain_order = ["Сухо", "Средне", "Дождливо"]
    valid = frame.dropna(subset=["temperature_2m_mean", "precipitation_sum", "metric"]).copy()
    if valid.empty:
        return valid
    temp_q = valid["temperature_2m_mean"].quantile([0.33, 0.66]).tolist()
    rain_q = valid["precipitation_sum"].quantile([0.33, 0.66]).tolist()
    valid["temp_regime"] = pd.cut(
        valid["temperature_2m_mean"],
        bins=[-np.inf, temp_q[0], temp_q[1], np.inf],
        labels=temp_order,
    )
    valid["rain_regime"] = pd.cut(
        valid["precipitation_sum"],
        bins=[-np.inf, rain_q[0], rain_q[1], np.inf],
        labels=rain_order,
    )
    valid["temp_regime"] = valid["temp_regime"].astype(str)
    valid["rain_regime"] = valid["rain_regime"].astype(str)
    return valid


def build_weather_effect_summary(returns_df: pd.DataFrame, metric: AnalysisMetric) -> Dict[str, object]:
    temp_order = ["Холодно", "Умеренно", "Тепло"]
    rain_order = ["Сухо", "Средне", "Дождливо"]
    valid = _build_weather_regimes(returns_df)
    if valid.empty:
        return {
            "available": False,
            "temp_order": temp_order,
            "rain_order": rain_order,
            "cells": [],
            "heatmap": [],
            "best_regime": "",
            "worst_regime": "",
            "best_minus_worst": float("nan"),
            "p_value": float("nan"),
            "metric": metric.value,
        }

    grouped = (
        valid.groupby(["temp_regime", "rain_regime"], as_index=False)["metric"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_metric", "count": "observations"})
    )

    cells: List[Dict[str, object]] = []
    for t in temp_order:
        for r in rain_order:
            row = grouped[(grouped["temp_regime"] == t) & (grouped["rain_regime"] == r)]
            if row.empty:
                cells.append(
                    {
                        "temp_regime": t,
                        "rain_regime": r,
                        "mean_metric": float("nan"),
                        "observations": 0,
                    }
                )
            else:
                cells.append(
                    {
                        "temp_regime": t,
                        "rain_regime": r,
                        "mean_metric": _safe_float(row.iloc[0]["mean_metric"]),
                        "observations": int(row.iloc[0]["observations"]),
                    }
                )

    heatmap = []
    for t in temp_order:
        line = []
        for r in rain_order:
            match = next(
                (
                    c["mean_metric"]
                    for c in cells
                    if c["temp_regime"] == t and c["rain_regime"] == r
                ),
                float("nan"),
            )
            line.append(match)
        heatmap.append(line)

    observed_cells = [c for c in cells if c["observations"] > 0 and not math.isnan(_safe_float(c["mean_metric"]))]
    best_regime = ""
    worst_regime = ""
    best_minus_worst = float("nan")
    p_value = float("nan")
    if len(observed_cells) >= 2:
        best = max(observed_cells, key=lambda x: _safe_float(x["mean_metric"]))
        worst = min(observed_cells, key=lambda x: _safe_float(x["mean_metric"]))
        best_regime = f"{best['temp_regime']} / {best['rain_regime']}"
        worst_regime = f"{worst['temp_regime']} / {worst['rain_regime']}"
        best_minus_worst = _safe_float(best["mean_metric"]) - _safe_float(worst["mean_metric"])
        best_values = valid[
            (valid["temp_regime"] == best["temp_regime"]) & (valid["rain_regime"] == best["rain_regime"])
        ]["metric"].astype(float).to_numpy()
        worst_values = valid[
            (valid["temp_regime"] == worst["temp_regime"]) & (valid["rain_regime"] == worst["rain_regime"])
        ]["metric"].astype(float).to_numpy()
        test = permutation_test_diff_means(best_values, worst_values)
        p_value = test.p_value

    return {
        "available": True,
        "temp_order": temp_order,
        "rain_order": rain_order,
        "cells": cells,
        "heatmap": heatmap,
        "best_regime": best_regime,
        "worst_regime": worst_regime,
        "best_minus_worst": best_minus_worst,
        "p_value": p_value,
        "metric": metric.value,
    }
