from __future__ import annotations

import logging
from datetime import date, timedelta

from flask import Flask, render_template, request

from analysis_core import METRIC_LABELS, AnalysisMetric, run_analysis

app = Flask(__name__)
logger = logging.getLogger("moex_web")


def _fmt_pct(value: float) -> str:
    return f"{value:.3%}"


def _fmt_volume(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} млрд ₽"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f} млн ₽"
    return f"{value:,.0f} ₽".replace(",", " ")


def _fmt_metric(value: float, metric: AnalysisMetric) -> str:
    if metric == AnalysisMetric.VOLUME:
        return _fmt_volume(value)
    return _fmt_pct(value)


@app.route("/", methods=["GET"])
def index() -> str:
    today = date.today()
    default_start = today - timedelta(days=365 * 3)
    start_raw = request.args.get("start", default_start.isoformat())
    end_raw = request.args.get("end", today.isoformat())
    max_tickers = int(request.args.get("max_tickers", 10))
    metric_raw = request.args.get("metric", AnalysisMetric.RETURN.value)
    try:
        metric = AnalysisMetric(metric_raw)
    except ValueError:
        metric = AnalysisMetric.RETURN
    run_calc = request.args.get("run", "0") == "1"

    try:
        start = date.fromisoformat(start_raw)
        end = date.fromisoformat(end_raw)
    except ValueError:
        start = default_start
        end = today

    context = {
        "params": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "max_tickers": max_tickers,
            "metric": metric.value,
        },
        "metric_options": [
            {"value": m.value, "label": METRIC_LABELS[m]} for m in AnalysisMetric
        ],
        "has_data": False,
        "error_message": "",
        "intro_message": "Нажмите 'Пересчитать', чтобы запустить анализ.",
    }

    if start >= end:
        context["error_message"] = "Дата начала должна быть раньше даты окончания."
        context["intro_message"] = ""
        return render_template("index.html", **context)

    if not run_calc:
        return render_template("index.html", **context)

    try:
        result = run_analysis(
            start=start, end=end, max_tickers=max_tickers, metric=metric, logger=logger
        )
    except Exception as exc:  # noqa: BLE001
        context["error_message"] = f"Ошибка расчета: {exc}"
        context["intro_message"] = ""
        return render_template("index.html", **context)

    lunar_table = result.lunar_stats.copy()
    lunar_table["mean_metric"] = lunar_table["mean_metric"].map(
        lambda v: _fmt_metric(v, result.metric)
    )
    lunar_table["median_metric"] = lunar_table["median_metric"].map(
        lambda v: _fmt_metric(v, result.metric)
    )

    weather_table = result.weather_stats.copy()
    if not weather_table.empty:
        weather_table["avg_metric"] = weather_table["avg_metric"].map(
            lambda v: _fmt_metric(v, result.metric)
        )

    halloween_table = result.halloween_by_ticker.copy()
    for col in [
        "winter_avg_daily_return",
        "summer_avg_daily_return",
        "avg_daily_return_diff",
        "winter_total_return",
        "summer_total_return",
        "total_return_diff",
    ]:
        halloween_table[col] = halloween_table[col].map(lambda v: _fmt_metric(v, result.metric))

    lunar_raw = result.lunar_stats.copy()
    weather_raw = result.weather_stats.copy()
    halloween_raw = result.halloween_by_ticker.copy()

    ticker_options = sorted(result.returns_df["ticker"].dropna().unique().tolist())
    phase_options = ["New Moon", "Waxing", "Full Moon", "Waning"]
    temp_options = (
        sorted(weather_raw["temp_regime"].dropna().astype(str).unique().tolist())
        if not weather_raw.empty
        else []
    )
    rain_options = (
        sorted(weather_raw["rain_regime"].dropna().astype(str).unique().tolist())
        if not weather_raw.empty
        else []
    )

    context.update(
        {
            "has_data": True,
            "intro_message": "",
            "tickers": ", ".join(result.tickers),
            "network_failures": result.network_failures,
            "metric_label": METRIC_LABELS[result.metric],
            "metric_kind": result.metric.value,
            "metrics": {
                "observations": f"{len(result.returns_df):,}".replace(",", " "),
                "mean": _fmt_metric(float(result.returns_df["metric"].mean()), result.metric),
                "median": _fmt_metric(float(result.returns_df["metric"].median()), result.metric),
            },
            "lunar_table": lunar_table.to_dict(orient="records"),
            "weather_table": weather_table.to_dict(orient="records"),
            "halloween_table": halloween_table.to_dict(orient="records"),
            "lunar_raw": lunar_raw.to_dict(orient="records"),
            "weather_raw": weather_raw.to_dict(orient="records"),
            "halloween_raw": halloween_raw.to_dict(orient="records"),
            "ticker_options": ticker_options,
            "phase_options": phase_options,
            "temp_options": temp_options,
            "rain_options": rain_options,
        }
    )
    return render_template("index.html", **context)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    app.run(host="0.0.0.0", port=5000, debug=True)
