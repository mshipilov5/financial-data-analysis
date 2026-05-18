from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from functools import lru_cache
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from astral import moon
from requests import Response
from requests.exceptions import RequestException, SSLError

MOEX_BASE = "https://iss.moex.com/iss"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MOSCOW_LAT = 55.7558
MOSCOW_LON = 37.6173
class AnalysisMetric(str, Enum):
    RETURN = "return"
    VOLATILITY = "volatility"
    VOLUME = "volume"


METRIC_LABELS = {
    AnalysisMetric.RETURN: "Доходность",
    AnalysisMetric.VOLATILITY: "Волатильность",
    AnalysisMetric.VOLUME: "Объём торгов",
}

VOLATILITY_WINDOW = 10

DEFAULT_LIQUID_POOL = [
    "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "SNGS", "SNGSP", "MOEX",
    "GMKN", "VTBR", "CHMF", "MAGN", "ALRS", "MTSS", "IRAO", "PHOR", "AFLT",
    "RUAL", "PLZL", "YDEX", "PIKK", "NLMK", "TRNFP", "RTKM", "BSPB", "CBOM",
    "OZON", "POSI", "BELU",
]


@dataclass
class HalloweenStats:
    winter_avg_daily_return: float
    summer_avg_daily_return: float
    winter_total_return: float
    summer_total_return: float


@dataclass
class HalloweenEffect:
    avg_daily_return_diff: float
    total_return_diff: float


@dataclass
class AnalysisResult:
    tickers: List[str]
    metric: AnalysisMetric
    network_failures: int
    returns_df: pd.DataFrame
    lunar_stats: pd.DataFrame
    weather_stats: pd.DataFrame
    halloween_by_ticker: pd.DataFrame


def parse_analysis_metric(value: str) -> AnalysisMetric:
    try:
        return AnalysisMetric(value.lower())
    except ValueError as exc:
        allowed = ", ".join(m.value for m in AnalysisMetric)
        raise ValueError(f"Unknown metric '{value}'. Use one of: {allowed}") from exc


def _extract_table(payload: dict, key: str) -> pd.DataFrame:
    table = payload.get(key, {})
    columns = table.get("columns", [])
    data = table.get("data", [])
    return pd.DataFrame(data, columns=columns)


def _get_json_with_fallback(
    url: str, params: dict, timeout: int = 20, logger: Optional[logging.Logger] = None
) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if logger:
                logger.debug("Request %s (attempt %s)", url, attempt + 1)
            response: Response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except SSLError as error:
            last_error = error
            if url.startswith("https://"):
                fallback_url = "http://" + url[len("https://") :]
                if logger:
                    logger.warning("SSL issue, fallback to HTTP: %s", fallback_url)
                try:
                    response = requests.get(fallback_url, params=params, timeout=timeout)
                    response.raise_for_status()
                    return response.json()
                except RequestException as fallback_error:
                    last_error = fallback_error
        except RequestException as error:
            last_error = error
            if logger:
                logger.warning("Network request failed: %s", error)
        timeout = min(timeout + attempt + 1, 35)
    if last_error:
        raise last_error
    raise RuntimeError("Unknown network error while requesting JSON.")


@lru_cache(maxsize=1)
def load_moex_securities(logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    if logger:
        logger.info("Loading MOEX securities list")
    url = f"{MOEX_BASE}/engines/stock/markets/shares/boards/TQBR/securities.json"
    params = {
        "securities.columns": "SECID,SHORTNAME,LISTLEVEL",
        "iss.meta": "off",
        "iss.only": "securities",
    }
    payload = _get_json_with_fallback(url, params=params, timeout=20, logger=logger)
    securities = _extract_table(payload, "securities")
    if securities.empty:
        return securities
    securities.columns = [str(col).upper() for col in securities.columns]
    if "LISTLEVEL" in securities.columns:
        securities = securities[securities["LISTLEVEL"].astype(str).isin(["1", "2"])].copy()
    securities = securities[[c for c in ["SECID", "SHORTNAME", "LISTLEVEL"] if c in securities.columns]]
    if "SECID" in securities.columns:
        securities = securities.sort_values(["LISTLEVEL", "SECID"], na_position="last")
    if logger:
        logger.info("Loaded %s securities (levels 1-2)", len(securities))
    return securities.reset_index(drop=True)


def load_moex_candles(
    secid: str, start: date, end: date, logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, bool]:
    url = f"{MOEX_BASE}/engines/stock/markets/shares/boards/TQBR/securities/{secid}/candles.json"
    params = {
        "from": start.isoformat(),
        "till": end.isoformat(),
        "interval": 24,
        "iss.meta": "off",
        "iss.only": "candles",
    }
    frames: List[pd.DataFrame] = []
    start_index = 0
    had_network_error = False
    while True:
        paged = dict(params)
        paged["start"] = start_index
        try:
            payload = _get_json_with_fallback(url, params=paged, timeout=20, logger=logger)
        except RequestException:
            had_network_error = True
            if logger:
                logger.warning("Skipping %s due to network error", secid)
            break
        block = _extract_table(payload, "candles")
        if block.empty:
            break
        frames.append(block)
        if len(block) < 100:
            break
        start_index += len(block)
    if not frames:
        return pd.DataFrame(), had_network_error
    candles = pd.concat(frames, ignore_index=True)
    candles["begin"] = pd.to_datetime(candles["begin"]).dt.date
    candles["close"] = pd.to_numeric(candles["close"], errors="coerce")
    candles["value"] = pd.to_numeric(candles.get("value"), errors="coerce")
    candles = candles.dropna(subset=["close"]).copy()
    candles = candles.rename(columns={"begin": "date"})
    return candles[["date", "close", "value"]], had_network_error


def load_moscow_weather(start: date, end: date, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    if logger:
        logger.info("Loading Moscow weather history")
    params = {
        "latitude": MOSCOW_LAT,
        "longitude": MOSCOW_LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Europe/Moscow",
    }
    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=25)
        response.raise_for_status()
        daily = response.json().get("daily", {})
    except RequestException:
        if logger:
            logger.warning("Weather API unavailable, analysis continues without weather block")
        return pd.DataFrame()
    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(daily.get("time", [])).date,
            "temperature_2m_mean": daily.get("temperature_2m_mean", []),
            "precipitation_sum": daily.get("precipitation_sum", []),
        }
    )
    if weather.empty:
        return weather
    weather["temperature_2m_mean"] = pd.to_numeric(weather["temperature_2m_mean"], errors="coerce")
    weather["precipitation_sum"] = pd.to_numeric(weather["precipitation_sum"], errors="coerce")
    return weather


def _attach_metric(candles: pd.DataFrame, metric: AnalysisMetric) -> pd.DataFrame:
    candles = candles.sort_values("date").copy()
    candles["daily_return"] = candles["close"].pct_change()
    if metric == AnalysisMetric.RETURN:
        candles["metric"] = candles["daily_return"]
    elif metric == AnalysisMetric.VOLATILITY:
        candles["metric"] = candles["daily_return"].rolling(VOLATILITY_WINDOW).std()
    else:
        candles["metric"] = candles["value"]
    return candles.dropna(subset=["metric"])


def build_market_dataset(
    secids: List[str],
    start: date,
    end: date,
    metric: AnalysisMetric = AnalysisMetric.RETURN,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, int]:
    parts: List[pd.DataFrame] = []
    failed_count = 0
    if logger:
        logger.info("Loading candles for %s tickers (%s)", len(secids), METRIC_LABELS[metric])
    for index, secid in enumerate(secids, start=1):
        if logger:
            logger.info("(%s/%s) %s", index, len(secids), secid)
        candles, had_network_error = load_moex_candles(secid, start, end, logger=logger)
        if had_network_error:
            failed_count += 1
        if candles.empty:
            continue
        candles["ticker"] = secid
        candles = _attach_metric(candles, metric)
        if candles.empty:
            continue
        parts.append(candles)
    if not parts:
        return pd.DataFrame(), failed_count
    return pd.concat(parts, ignore_index=True), failed_count


def build_returns_dataset(
    secids: List[str], start: date, end: date, logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, int]:
    return build_market_dataset(secids, start, end, AnalysisMetric.RETURN, logger=logger)


def select_tickers(securities: pd.DataFrame, max_tickers: int) -> List[str]:
    available = securities["SECID"].tolist()
    available_set = set(available)
    candidate = [t for t in DEFAULT_LIQUID_POOL if t in available_set]
    if len(candidate) < max_tickers:
        candidate.extend([t for t in available if t not in set(candidate)][: max_tickers - len(candidate)])
    return candidate[:max_tickers]


def lunar_phase_label(current_date: date) -> str:
    phase = moon.phase(current_date)
    if phase < 3.5 or phase > 24.5:
        return "New Moon"
    if phase < 10.5:
        return "Waxing"
    if phase < 17.5:
        return "Full Moon"
    return "Waning"


def _season_aggregate(series: pd.Series, metric: AnalysisMetric, kind: Literal["avg", "total"]) -> float:
    if series.empty:
        return 0.0
    if kind == "avg":
        return float(series.mean())
    if metric == AnalysisMetric.RETURN:
        return float((1 + series).prod() - 1)
    if metric == AnalysisMetric.VOLUME:
        return float(series.sum())
    return float(series.mean())


def compute_halloween_stats(data: pd.DataFrame, metric: AnalysisMetric) -> HalloweenStats:
    tagged = data.copy()
    tagged["month"] = pd.to_datetime(tagged["date"]).dt.month
    tagged["season"] = np.where(
        tagged["month"].isin([11, 12, 1, 2, 3, 4]),
        "Winter (Nov-Apr)",
        "Summer (May-Oct)",
    )
    winter = tagged[tagged["season"] == "Winter (Nov-Apr)"]["metric"]
    summer = tagged[tagged["season"] == "Summer (May-Oct)"]["metric"]
    return HalloweenStats(
        winter_avg_daily_return=_season_aggregate(winter, metric, "avg"),
        summer_avg_daily_return=_season_aggregate(summer, metric, "avg"),
        winter_total_return=_season_aggregate(winter, metric, "total"),
        summer_total_return=_season_aggregate(summer, metric, "total"),
    )


def compute_halloween_effect(stats: HalloweenStats) -> HalloweenEffect:
    return HalloweenEffect(
        avg_daily_return_diff=stats.winter_avg_daily_return - stats.summer_avg_daily_return,
        total_return_diff=stats.winter_total_return - stats.summer_total_return,
    )


def compute_halloween_by_ticker(data: pd.DataFrame, metric: AnalysisMetric) -> pd.DataFrame:
    rows = []
    for ticker, frame in data.groupby("ticker"):
        stats = compute_halloween_stats(frame, metric)
        effect = compute_halloween_effect(stats)
        rows.append(
            {
                "ticker": ticker,
                "winter_avg_daily_return": stats.winter_avg_daily_return,
                "summer_avg_daily_return": stats.summer_avg_daily_return,
                "avg_daily_return_diff": effect.avg_daily_return_diff,
                "winter_total_return": stats.winter_total_return,
                "summer_total_return": stats.summer_total_return,
                "total_return_diff": effect.total_return_diff,
            }
        )
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def run_analysis(
    start: date,
    end: date,
    max_tickers: int,
    metric: AnalysisMetric | str = AnalysisMetric.RETURN,
    logger: Optional[logging.Logger] = None,
) -> AnalysisResult:
    if isinstance(metric, str):
        metric = parse_analysis_metric(metric)
    max_tickers = max(5, min(max_tickers, 20))
    securities = load_moex_securities(logger=logger)
    if securities.empty:
        raise RuntimeError("No securities loaded from MOEX")

    tickers = select_tickers(securities, max_tickers)
    if not tickers:
        raise RuntimeError("No tickers selected for analysis")

    returns_df, network_failures = build_market_dataset(
        tickers, start, end, metric=metric, logger=logger
    )
    if returns_df.empty:
        raise RuntimeError(f"No data collected for metric '{metric.value}'")

    returns_df["phase"] = returns_df["date"].apply(lunar_phase_label)
    weather = load_moscow_weather(start, end, logger=logger)
    if weather.empty:
        returns_df["temperature_2m_mean"] = np.nan
        returns_df["precipitation_sum"] = np.nan
    else:
        returns_df = returns_df.merge(weather, on="date", how="left")

    lunar_stats = (
        returns_df.groupby(["ticker", "phase"], as_index=False)
        .agg(
            mean_metric=("metric", "mean"),
            median_metric=("metric", "median"),
            observations=("metric", "count"),
        )
        .sort_values(["ticker", "phase"])
        .reset_index(drop=True)
    )
    lunar_stats["phase"] = pd.Categorical(
        lunar_stats["phase"],
        categories=["New Moon", "Waxing", "Full Moon", "Waning"],
        ordered=True,
    )
    lunar_stats = lunar_stats.sort_values(["ticker", "phase"]).reset_index(drop=True)

    valid_weather = returns_df.dropna(subset=["temperature_2m_mean", "precipitation_sum"]).copy()
    weather_stats = pd.DataFrame(columns=["ticker", "temp_regime", "rain_regime", "avg_metric"])
    if not valid_weather.empty:
        temp_q = valid_weather["temperature_2m_mean"].quantile([0.33, 0.66]).tolist()
        rain_q = valid_weather["precipitation_sum"].quantile([0.33, 0.66]).tolist()
        valid_weather["temp_regime"] = pd.cut(
            valid_weather["temperature_2m_mean"],
            bins=[-np.inf, temp_q[0], temp_q[1], np.inf],
            labels=["Холодно", "Умеренно", "Тепло"],
        )
        valid_weather["rain_regime"] = pd.cut(
            valid_weather["precipitation_sum"],
            bins=[-np.inf, rain_q[0], rain_q[1], np.inf],
            labels=["Сухо", "Средне", "Дождливо"],
        )
        weather_stats = (
            valid_weather.groupby(["ticker", "temp_regime", "rain_regime"], as_index=False)["metric"]
            .mean()
            .rename(columns={"metric": "avg_metric"})
            .sort_values(["ticker", "temp_regime", "rain_regime"])
            .reset_index(drop=True)
        )

    halloween_by_ticker = compute_halloween_by_ticker(returns_df, metric)

    return AnalysisResult(
        tickers=tickers,
        metric=metric,
        network_failures=network_failures,
        returns_df=returns_df,
        lunar_stats=lunar_stats,
        weather_stats=weather_stats,
        halloween_by_ticker=halloween_by_ticker,
    )
