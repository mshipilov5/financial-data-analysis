from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from analysis_core import run_analysis

logger = logging.getLogger("moex_cli")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    today = date.today()
    default_start = today - timedelta(days=365 * 3)
    parser = argparse.ArgumentParser(
        description="CLI-анализ акций Мосбиржи (по каждой акции отдельно)."
    )
    parser.add_argument("--start", default=default_start.isoformat(), help="Дата начала YYYY-MM-DD")
    parser.add_argument("--end", default=today.isoformat(), help="Дата конца YYYY-MM-DD")
    parser.add_argument("--max-tickers", type=int, default=10, help="Количество тикеров (5-20)")
    parser.add_argument("--output-dir", default="output", help="Папка для CSV-отчетов")
    parser.add_argument("--verbose", action="store_true", help="Включить DEBUG-логи")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        return 1
    if start >= end:
        logger.error("Start date must be earlier than end date")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Start analysis: %s -> %s", start, end)

    try:
        result = run_analysis(start=start, end=end, max_tickers=args.max_tickers, logger=logger)
    except Exception as exc:  # noqa: BLE001
        logger.error("Analysis failed: %s", exc)
        return 1

    returns_path = output_dir / "returns.csv"
    lunar_path = output_dir / "lunar_stats.csv"
    weather_path = output_dir / "weather_stats.csv"
    halloween_path = output_dir / "halloween_effect_by_ticker.csv"

    result.returns_df.to_csv(returns_path, index=False)
    result.lunar_stats.to_csv(lunar_path, index=False)
    result.weather_stats.to_csv(weather_path, index=False)
    result.halloween_by_ticker.to_csv(halloween_path, index=False)

    logger.info("Selected tickers: %s", ", ".join(result.tickers))
    logger.info("Observations: %s", len(result.returns_df))
    logger.info("Mean daily return: %.3f%%", result.returns_df["daily_return"].mean() * 100)
    logger.info("Median daily return: %.3f%%", result.returns_df["daily_return"].median() * 100)
    logger.info("Network failures while loading candles: %s", result.network_failures)
    logger.info("Saved: %s, %s, %s, %s", returns_path, lunar_path, weather_path, halloween_path)

    print("\nLunar stats by ticker:")
    print(result.lunar_stats.to_string(index=False))
    if result.weather_stats.empty:
        print("\nWeather stats: no data for selected period.")
    else:
        print("\nWeather stats by ticker:")
        print(result.weather_stats.to_string(index=False))
    print("\nHalloween effect by ticker:")
    print(result.halloween_by_ticker.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
