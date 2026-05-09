# Financial Data Analysis (MOEX)

This project analyzes Moscow Exchange (MOEX) stocks using MOEX ISS API data and `pandas`.

It supports two interfaces:
- **CLI application** for terminal-based analysis and CSV export
- **Web application** (Flask) with interactive filters and charts

Both interfaces use the same shared analytics module and compute metrics **per ticker**.

## Features

- Loads level 1-2 Russian stocks traded on MOEX (`TQBR`)
- Fetches daily candles and computes daily returns
- Evaluates non-traditional strategy signals:
  - Lunar phases
  - Weather conditions in Moscow
  - Halloween effect (`Nov-Apr` vs `May-Oct`)
- Handles unstable SSL/network behavior with retries and HTTPS/HTTP fallback

## Project Structure

- `analysis_core.py` - shared data loading and analytics logic
- `cli_app.py` - CLI entrypoint (logs + CSV export)
- `web_app.py` - Flask web entrypoint
- `templates/index.html` - web UI with checkbox filters and charts
- `app.py` - backward-compatible alias for CLI (`python app.py`)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run CLI

Basic run:

```bash
python cli_app.py
```

Example with custom parameters:

```bash
python cli_app.py --start 2024-01-01 --end 2026-01-01 --max-tickers 10 --verbose
```

CLI arguments:
- `--start` start date in `YYYY-MM-DD`
- `--end` end date in `YYYY-MM-DD`
- `--max-tickers` number of tickers to analyze (clamped to `5..20`)
- `--output-dir` output folder for CSV files (default: `output`)
- `--verbose` enables debug logs

## Run Web App

```bash
python web_app.py
```

Open in browser:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

The web page includes:
- Checkbox filters for ticker, moon phase, temperature regime, and precipitation regime
- Dynamic tables
- Interactive Plotly charts that update based on selected filters

## Output Files (CLI)

Saved to `output/` (or your custom `--output-dir`):
- `returns.csv`
- `lunar_stats.csv`
- `weather_stats.csv`
- `halloween_effect_by_ticker.csv`
