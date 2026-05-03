"""EverestAPI CLI — command-line interface for the EverestQuant tournament platform.

Usage:
    everestapi health
    everestapi download-data --universe futures --split train
    everestapi submit --model my-model --file predictions.csv
    everestapi leaderboard --period 30d
    everestapi scores --model my-model --days 30
    everestapi rounds --tournament equities
    everestapi credits
    everestapi register --name my-agent --email me@example.com
"""

from __future__ import annotations

import json
import sys

import click

from everestapi import EverestAPI, EverestError, __version__


def _get_api(api_key: str | None = None) -> EverestAPI:
    return EverestAPI(api_key=api_key or "")


def _print_json(data: dict) -> None:
    click.echo(json.dumps(data, indent=2))


@click.group()
@click.version_option(__version__, prog_name="everestapi")
def cli() -> None:
    """EverestQuant tournament CLI."""


@cli.command()
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def health(api_key: str | None) -> None:
    """Check API health."""
    api = _get_api(api_key)
    try:
        _print_json(api.health())
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("download-data")
@click.option("--universe", "-u", default="futures", help="Universe (futures/equities)")
@click.option("--split", "-s", default="train", help="Split (train/validation/test/live)")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def download_data(universe: str, split: str, output: str | None, api_key: str | None) -> None:
    """Download tournament dataset."""
    api = _get_api(api_key)
    try:
        path = api.download_dataset(universe=universe, split=split, output_path=output)
        click.echo(f"Downloaded to {path}")
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("download-benchmark")
@click.option("--universe", "-u", default="futures")
@click.option("--split", "-s", default="validation")
@click.option("--version", "-v", default="v0")
@click.option("--output", "-o", default=None)
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def download_benchmark(universe: str, split: str, version: str, output: str | None, api_key: str | None) -> None:
    """Download benchmark model predictions."""
    api = _get_api(api_key)
    try:
        path = api.download_benchmark(universe=universe, split=split, version=version, output_path=output)
        click.echo(f"Downloaded to {path}")
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", required=True, help="Model identifier")
@click.option("--file", "-f", "file_path", required=True, help="CSV file with predictions")
@click.option("--tournament", "-t", default="equities", help="Tournament (equities/futures)")
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def submit(model: str, file_path: str, tournament: str, api_key: str | None) -> None:
    """Submit predictions from a CSV file."""
    import csv

    api = _get_api(api_key)
    try:
        with open(file_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if tournament == "futures":
            preds = {row["instrument_id"]: float(row["prediction"]) for row in rows}
            result = api.submit_futures_predictions(model_id=model, predictions=preds)
        else:
            preds_list = [{"ticker": row["ticker"], "score": float(row["score"])} for row in rows]
            result = api.submit_predictions(model_id=model, predictions=preds_list)

        _print_json(result)
    except (EverestError, KeyError, FileNotFoundError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--period", "-p", default="30d", help="Period (7d/30d/90d/all)")
@click.option("--tournament", "-t", default="equities", help="Tournament (equities/futures)")
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def leaderboard(period: str, tournament: str, api_key: str | None) -> None:
    """Show tournament leaderboard."""
    api = _get_api(api_key)
    try:
        if tournament == "futures":
            data = api.get_futures_leaderboard(period=period)
        else:
            data = api.get_leaderboard(period=period)
        _print_json(data)
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", required=True, help="Model identifier")
@click.option("--days", "-d", default=30, help="Look-back window")
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def scores(model: str, days: int, api_key: str | None) -> None:
    """Show scoring results for a model."""
    api = _get_api(api_key)
    try:
        _print_json(api.get_scores(model_id=model, days=days))
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", required=True, help="Model identifier")
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def diagnostics(model: str, api_key: str | None) -> None:
    """Show per-round diagnostics for a model."""
    api = _get_api(api_key)
    try:
        _print_json(api.get_round_diagnostics(model_id=model))
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--tournament", "-t", default="equities", help="Tournament (equities/futures)")
@click.option("--limit", "-n", default=10, help="Number of rounds")
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def rounds(tournament: str, limit: int, api_key: str | None) -> None:
    """List tournament rounds."""
    api = _get_api(api_key)
    try:
        _print_json(api.get_rounds(tournament=tournament, limit=limit))
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def credits(api_key: str | None) -> None:
    """Check compute credit balance."""
    api = _get_api(api_key)
    try:
        _print_json(api.get_compute_credits())
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", required=True, help="Model identifier")
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def stake_balance(model: str, api_key: str | None) -> None:
    """Show stake balance and pending payouts."""
    api = _get_api(api_key)
    try:
        _print_json(api.get_stake_balance(model_id=model))
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--name", "-n", required=True, help="Agent name")
@click.option("--email", "-e", required=True, help="Recovery email")
def register(name: str, email: str) -> None:
    """Register a new agent (no API key needed)."""
    api = _get_api()
    try:
        result = api.register(name=name, email=email)
        click.echo("Registration successful!")
        click.echo(f"  Agent ID:  {result['agent_id']}")
        click.echo(f"  API Key:   {result['api_key']}  (save this — shown once)")
        click.echo(f"\nSet your API key:")
        click.echo(f"  export EVEREST_API_KEY={result['api_key']}")
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--api-key", envvar="EVEREST_API_KEY", default=None)
def universe(api_key: str | None) -> None:
    """Show current tournament universe."""
    api = _get_api(api_key)
    try:
        _print_json(api.get_universe())
    except EverestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
