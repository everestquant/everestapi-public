"""EverestAPI — Python client for the EverestQuant prediction tournament platform.

Usage::

    from everestapi import EverestAPI

    api = EverestAPI(api_key="eiq_your_key")

    # Get universe
    universe = api.get_universe()

    # Get features for today
    features = api.get_features(date="2026-04-02")

    # Submit predictions
    predictions = [{"ticker": "AAPL", "score": 0.5}, ...]
    result = api.submit_predictions(model_id="my-model", predictions=predictions)

    # Get scores
    scores = api.get_scores(model_id="my-model", days=30)

    # Get leaderboard
    leaderboard = api.get_leaderboard(period="30d")
"""

from __future__ import annotations

import os
from pathlib import PurePath
from typing import Any

import httpx


def _safe_output_path(output_path: str, *, server_supplied: bool = False) -> str:
    """Validate a download destination path.

    Rejects any path that would escape the intended directory via ``..``
    segments (after normalisation). When ``server_supplied`` is True the
    caller is treating ``output_path`` as a filename returned by the server
    and we additionally strip directory components defensively via
    :func:`os.path.basename`.
    """
    if server_supplied:
        # Server-supplied filename: only keep the basename, never honour any
        # directory components the server tries to inject.
        output_path = os.path.basename(output_path)
        if not output_path or output_path in (".", ".."):
            raise ValueError(
                f"Refusing to write to invalid server-supplied filename: {output_path!r}",
            )
        return output_path

    # User-supplied path. Absolute paths are fine; reject any '..' segments
    # that survive normalisation (these would let a path traversal escape
    # the user's intended directory).
    parts = PurePath(output_path).parts
    if any(part == ".." for part in parts):
        raise ValueError(
            f"Refusing to write to path containing '..' segments: {output_path!r}",
        )
    return output_path


def _open_no_symlink(path: str):
    """Open ``path`` for binary write, refusing to follow a pre-existing symlink.

    Cross-platform shim:
      * On POSIX, uses ``os.open`` with ``O_NOFOLLOW`` so opening a symlink
        fails atomically.
      * On Windows (no ``O_NOFOLLOW``), uses ``os.lstat`` to detect a symlink
        before opening. There is a small TOCTOU window on Windows but it
        materially raises the bar over a plain ``open``.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC | nofollow
        fd = os.open(path, flags, 0o644)
        return os.fdopen(fd, "wb")

    # Windows path: pre-flight lstat check. If the target exists and is a
    # symlink, refuse to open it.
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        st = None
    if st is not None:
        import stat as _stat
        if _stat.S_ISLNK(st.st_mode):
            raise OSError(
                f"Refusing to write to symlink at {path!r} (symlink-following blocked)",
            )
    return open(path, "wb")


class EverestError(Exception):
    """Error returned by the EverestQuant API."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"EverestQuant API error {status_code}: {detail}")


class EverestAPI:
    """Synchronous client for the EverestQuant tournament API.

    Parameters
    ----------
    api_key : str, optional
        EverestQuant API key. Falls back to ``EVEREST_API_KEY`` env var.
    base_url : str, optional
        Base URL for the API. Defaults to ``https://everestquant.ai`` (apex
        domain), overridable via ``EVEREST_API_URL`` env var.
    timeout : float
        HTTP timeout in seconds.
    tournament : str
        Default tournament ("equities" or "futures").
    basic_auth : tuple[str, str], optional
        While the platform is in pre-launch lockdown, set
        ``basic_auth=('user', 'pass')`` to satisfy the Basic Auth gate.
        Remove once the platform is publicly launched. Falls back to the
        ``EIQ_BASIC_AUTH_USER`` / ``EIQ_BASIC_AUTH_PASS`` env vars when not
        explicitly supplied.
    """

    DEFAULT_BASE_URL = os.getenv("EVEREST_API_URL", "https://everestquant.ai")

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        tournament: str = "equities",
        basic_auth: tuple[str, str] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("EVEREST_API_KEY", "")
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.tournament = tournament

        # Basic Auth fallback: env vars only used if both user and pass are set.
        if basic_auth is None:
            env_user = os.getenv("EIQ_BASIC_AUTH_USER")
            env_pass = os.getenv("EIQ_BASIC_AUTH_PASS")
            if env_user and env_pass:
                basic_auth = (env_user, env_pass)
        self.basic_auth = basic_auth

        headers: dict[str, str] = {}
        if self.api_key:
            # Only inject the X-API-Key header when we actually have one;
            # an empty header value is meaningless and may show up in
            # server logs.
            headers["X-API-Key"] = self.api_key

        client_kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "headers": headers,
            "timeout": timeout,
        }
        if self.basic_auth is not None:
            client_kwargs["auth"] = self.basic_auth
        self._client = httpx.Client(**client_kwargs)

    def __repr__(self) -> str:
        return (
            f"<EverestAPI base_url={self.base_url!r} "
            f"api_key=*** authenticated={bool(self.api_key)}>"
        )

    # -- helpers ----------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        resp = self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise EverestError(resp.status_code, detail)
        return resp.json()

    def _download(self, path: str, output_path: str) -> str:
        resp = self._client.request("GET", path)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise EverestError(resp.status_code, detail)
        safe_path = _safe_output_path(output_path)
        with _open_no_symlink(safe_path) as f:
            f.write(resp.content)
        return safe_path

    # -- public API -------------------------------------------------------

    def health(self) -> dict:
        """GET /api/v1/health"""
        return self._request("GET", "/api/v1/health")

    def get_universe(self) -> dict:
        """GET /api/v1/universe — current tournament universe."""
        return self._request("GET", "/api/v1/universe")

    def get_features(self, date: str | None = None) -> dict:
        """GET /api/v1/features — obfuscated features for a date."""
        params: dict[str, str] = {}
        if date is not None:
            params["date"] = date
        return self._request("GET", "/api/v1/features", params=params)

    def submit_predictions(
        self,
        model_id: str,
        predictions: list[dict],
    ) -> dict:
        """POST /api/v1/predictions — submit daily predictions.

        Parameters
        ----------
        model_id : str
            Unique model identifier.
        predictions : list[dict]
            Each dict must have ``ticker`` (str) and ``score`` (float in [-1, 1]).
        """
        body = {"model_id": model_id, "predictions": predictions}
        return self._request("POST", "/api/v1/predictions", json=body)

    def submit_predictions_file(
        self,
        model_name: str,
        file_path: str,
        tournament: str | None = None,
    ) -> dict:
        """POST /api/v1/predictions/upload — submit predictions as a CSV or Parquet file.

        Parameters
        ----------
        model_name : str
            Model identifier.
        file_path : str
            Path to a ``.csv`` or ``.parquet`` file with columns ``ticker`` (str)
            and ``score`` (float in [-1, 1]).
        tournament : str, optional
            Tournament identifier. Defaults to the client's tournament.
        """
        tourn = tournament or self.tournament
        with open(file_path, "rb") as f:
            fname = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            resp = self._client.post(
                "/api/v1/predictions/upload",
                files={"file": (fname, f, "application/octet-stream")},
                params={"model_id": model_name, "tournament": tourn},
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise EverestError(resp.status_code, detail)
        return resp.json()

    def submit_validation_diagnostics(
        self,
        model_id: str,
        predictions,
        tournament: str = "futures",
        target: str = "target_everest_20",
    ) -> dict:
        """POST /api/v1/diagnostics/upload — score predictions against validation data."""
        if hasattr(predictions, "to_dict"):
            preds_list = predictions.to_dict(orient="records")
        else:
            preds_list = list(predictions)
        body = {
            "model_id": model_id,
            "tournament": tournament,
            "target": target,
            "predictions": preds_list,
        }
        return self._request("POST", "/api/v1/diagnostics/upload", json=body)

    def get_validation_panel(self, model_id: str, days: int = 365) -> dict:
        """GET /api/v1/models/{model_id}/diagnostics/validation — full validation metrics."""
        return self._request(
            "GET",
            f"/api/v1/models/{model_id}/diagnostics/validation",
            params={"days": days},
        )

    def get_validation_diagnostics(self, model_id: str, days: int = 365) -> dict:
        """Alias for :meth:`get_validation_panel` — full validation diagnostics panel."""
        return self.get_validation_panel(model_id=model_id, days=days)

    def get_model_per_exped_breakdown(
        self, model_id: str, days: int = 3650,
    ) -> list[float]:
        """Return the per-exped CORR series for a model (oldest→newest).

        Thin convenience wrapper over :meth:`get_validation_diagnostics`.
        """
        resp = self.get_validation_diagnostics(model_id, days=days)
        return list(resp.get("per_exped_corr", []))

    def get_job_output(
        self, job_id: str, filename: str, output_path: str | None = None,
    ) -> str:
        """Download an output file from a completed compute job.

        .. note::
            PLANNED endpoint — server-side route
            ``GET /api/v1/compute/jobs/{job_id}/output/{filename}`` may return 404
            until the compute output-file infrastructure is deployed.
        """
        if output_path is None:
            # ``filename`` flows through to the URL but is also the default
            # destination on disk; sanitise as server-supplied so a value
            # like "../etc/passwd" can't escape cwd.
            output_path = _safe_output_path(filename, server_supplied=True)
        return self._download(
            f"/api/v1/compute/jobs/{job_id}/output/{filename}", output_path,
        )

    def get_scores(self, model_id: str, days: int = 30) -> dict:
        """GET /api/v1/scores — scoring results for a model."""
        params = {"model_id": model_id, "days": str(days)}
        return self._request("GET", "/api/v1/scores", params=params)

    def get_leaderboard(self, period: str = "30d") -> dict:
        """GET /api/v1/leaderboard — tournament leaderboard."""
        return self._request("GET", "/api/v1/leaderboard", params={"period": period})

    # -- data & diagnostics -----------------------------------------------

    def download_dataset(
        self,
        universe: str = "futures",
        split: str = "train",
        output_path: str | None = None,
        version: str = "v0",
    ) -> str:
        """Download a parquet dataset to a local file."""
        if output_path is None:
            output_path = f"{universe}_{split}.parquet"
        return self._download(f"/api/v1/data/download/{universe}/{split}", output_path)

    def get_dataset_info(self, universe: str = "futures") -> dict:
        """Get metadata about a dataset universe."""
        return self._request("GET", f"/api/v1/data/info/{universe}")

    def get_diagnostics(self, model_id: str, date: str | None = None) -> dict:
        """Get scoring diagnostics for a model."""
        params: dict[str, str] = {"model_id": model_id}
        if date is not None:
            params["date"] = date
        return self._request("GET", f"/api/v1/diagnostics/{model_id}", params=params)

    def get_round_diagnostics(self, model_id: str) -> dict:
        """GET /api/v1/diagnostics/{model_id}/rounds — per-round scoring breakdown."""
        return self._request("GET", f"/api/v1/diagnostics/{model_id}/rounds")

    # -- futures tournament -----------------------------------------------

    def get_futures_universe(self) -> dict:
        """GET /api/v1/futures/universe — obfuscated futures universe."""
        return self._request("GET", "/api/v1/futures/universe")

    def get_futures_targets(self) -> dict:
        """GET /api/v1/futures/targets — list available target names."""
        return self._request("GET", "/api/v1/futures/targets")

    def submit_futures_predictions(
        self,
        model_id: str,
        predictions: dict[str, float],
        exped: str | None = None,
    ) -> dict:
        """POST /api/v1/futures/submit/v2 — submit futures predictions.

        Parameters
        ----------
        model_id : str
            Model identifier.
        predictions : dict[str, float]
            {instrument_id: prediction} for the primary target.
        exped : str, optional
            Exped identifier. Defaults to current round.
        """
        body = {
            "model_id": model_id,
            "exped": exped or "current",
            "predictions": [
                {"instrument_id": iid, "prediction": pred}
                for iid, pred in predictions.items()
            ],
        }
        return self._request("POST", "/api/v1/futures/submit/v2", json=body)

    def submit_futures_predictions_legacy(
        self,
        model_id: str,
        exped: str,
        predictions: dict[str, dict[str, float]],
    ) -> dict:
        """POST /api/v1/futures/submit — submit per-target predictions (legacy v1 format).

        .. deprecated::
            Use :meth:`submit_futures_predictions` (v2) instead.
        """
        preds_list = [
            {"instrument_id": k, "targets": v}
            for k, v in sorted(predictions.items())
        ]
        body = {"model_id": model_id, "exped": exped, "predictions": preds_list}
        return self._request("POST", "/api/v1/futures/submit", json=body)

    def get_futures_leaderboard(self, period: str = "30d") -> dict:
        """GET /api/v1/futures/leaderboard — ranked by final_score."""
        return self._request("GET", "/api/v1/futures/leaderboard", params={"period": period})

    def get_futures_diagnostics(self, agent_id: str, round_id: int | None = None) -> dict:
        """GET /api/v1/futures/diagnostics/{agent_id}"""
        params = {}
        if round_id is not None:
            params["round_id"] = str(round_id)
        return self._request("GET", f"/api/v1/futures/diagnostics/{agent_id}", params=params)

    def download_futures_data(
        self,
        split: str = "train",
        output_path: str | None = None,
    ) -> str:
        """Download futures parquet dataset."""
        if output_path is None:
            output_path = f"futures_{split}.parquet"
        return self._download(f"/api/v1/futures/data/{split}", output_path)

    # -- data discovery ---------------------------------------------------

    def get_dataset_schema(self, version: str = "v0") -> dict:
        """Get features.json schema for a dataset version."""
        return self._request("GET", f"/api/v1/data/{version}/schema")

    def list_versions(self) -> dict:
        """List available dataset versions."""
        return self._request("GET", "/api/v1/data/versions")

    def download_benchmark(
        self,
        universe: str = "futures",
        split: str = "validation",
        version: str = "v0",
        output_path: str | None = None,
    ) -> str:
        """Download benchmark model predictions."""
        if output_path is None:
            output_path = f"benchmark_{universe}_{split}.parquet"
        return self._download(
            f"/api/v1/data/{version}/benchmark/{universe}/{split}", output_path,
        )

    def download_ai_model(
        self,
        universe: str = "futures",
        version: str = "v0",
        output_path: str | None = None,
    ) -> str:
        """Download stake-weighted ai-model predictions."""
        if output_path is None:
            output_path = f"ai_model_{universe}.parquet"
        return self._download(
            f"/api/v1/data/{version}/ai-model/{universe}", output_path,
        )

    # -- compute ----------------------------------------------------------

    def quick_train(
        self,
        model: str = "lightgbm",
        features: str = "small",
        target: str = "target_everest_20",
        universe: str = "futures",
        params: dict | None = None,
    ) -> dict:
        """Submit Tier 1 serverless training job."""
        body: dict = {
            "model": model, "features": features, "target": target,
            "universe": universe,
        }
        if params:
            body["params"] = params
        return self._request("POST", "/api/v1/compute/quick-train", json=body)

    def custom_train(
        self,
        script: str | None = None,
        script_path: str | None = None,
        files: dict[str, str] | None = None,
        gpu: str = "T4",
        max_hours: float = 1.0,
        requirements: list[str] | None = None,
    ) -> dict:
        """Submit Tier 2 custom training job."""
        if script_path and not script:
            with open(script_path) as f:
                script = f.read()
        if files and not script:
            parts = [f"# --- FILE: {name} ---\n{content}" for name, content in files.items()]
            script = "\n\n".join(parts)
        body: dict = {"script": script, "gpu": gpu, "max_hours": max_hours}
        if requirements:
            body["requirements"] = requirements
        return self._request("POST", "/api/v1/compute/custom-train", json=body)

    def get_job_status(self, job_id: str) -> dict:
        """Poll compute job status."""
        return self._request("GET", f"/api/v1/compute/jobs/{job_id}")

    def get_model_download_url(self, job_id: str) -> dict:
        """Get a presigned download URL for a trained model."""
        return self._request("GET", f"/api/v1/compute/jobs/{job_id}/model")

    def download_model(self, job_id: str, output_path: str | None = None) -> str:
        """Download the trained model pickle from a completed job."""
        info = self.get_model_download_url(job_id)
        url = info["download_url"]
        if output_path is None:
            output_path = f"{job_id}.pkl"
        safe_path = _safe_output_path(output_path)
        # Presigned S3 URLs reject our X-API-Key header signing, so we
        # don't reuse ``self._client``; instead use a fresh httpx.get with
        # an explicit timeout matching the rest of the SDK so a hung S3
        # endpoint can't block the caller indefinitely.
        resp = httpx.get(url, follow_redirects=True, timeout=self._client.timeout)
        resp.raise_for_status()
        with _open_no_symlink(safe_path) as f:
            f.write(resp.content)
        return safe_path

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a running compute job."""
        return self._request("DELETE", f"/api/v1/compute/jobs/{job_id}")

    def wait_for_job(self, job_id: str, timeout: int = 3600) -> dict:
        """Block until job completes. Exponential backoff polling."""
        import time
        delay = 2.0
        elapsed = 0.0
        while elapsed < timeout:
            status = self.get_job_status(job_id)
            if status["status"] in ("completed", "failed", "timeout", "cancelled"):
                return status
            time.sleep(delay)
            elapsed += delay
            delay = min(delay * 2, 30.0)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    def get_compute_credits(self) -> dict:
        """Check compute credit balance."""
        return self._request("GET", "/api/v1/compute/credits")

    def list_model_templates(self) -> dict:
        """List available quick-train model templates."""
        return self._request("GET", "/api/v1/compute/models")

    # -- registration (no auth) ------------------------------------------

    def register(self, name: str, email: str) -> dict:
        """POST /api/v1/agents/register — register a new agent (no auth needed).

        Returns dict with agent_id, name, api_key (shown once), webhook_secret.
        """
        resp = httpx.post(
            f"{self.base_url}/api/v1/agents/register",
            json={"name": name, "email": email},
            timeout=self._client.timeout,
            auth=self.basic_auth,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise EverestError(resp.status_code, detail)
        return resp.json()

    # -- rounds -----------------------------------------------------------

    def get_rounds(self, tournament: str = "equities", limit: int = 25) -> dict:
        """GET /api/v1/rounds — list tournament rounds."""
        return self._request("GET", "/api/v1/rounds", params={
            "tournament": tournament, "limit": str(limit),
        })

    def get_current_round(self, tournament: str = "equities") -> dict:
        """GET /api/v1/rounds/current — current active round."""
        return self._request("GET", "/api/v1/rounds/current", params={
            "tournament": tournament,
        })

    def get_schedule(self) -> dict:
        """GET /api/v1/schedule — round schedule for both tournaments."""
        return self._request("GET", "/api/v1/schedule")

    # -- model uploads ----------------------------------------------------

    def upload_model(self, model_id: str, file_path: str) -> dict:
        """POST /api/v1/models/{model_id}/upload — upload a .pkl model file."""
        with open(file_path, "rb") as f:
            resp = self._client.post(
                f"/api/v1/models/{model_id}/upload",
                files={"file": (f.name, f, "application/octet-stream")},
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise EverestError(resp.status_code, detail)
        return resp.json()

    def get_upload_status(self, model_id: str) -> dict:
        """GET /api/v1/models/{model_id}/upload/status"""
        return self._request("GET", f"/api/v1/models/{model_id}/upload/status")

    # -- multipliers ------------------------------------------------------

    def set_multipliers(self, model_id: str, corr_mult: float, aimc_mult: float) -> dict:
        """PATCH /api/v1/models/{model_id}/multipliers — set payout multipliers."""
        return self._request("PATCH", f"/api/v1/models/{model_id}/multipliers", json={
            "corr_multiplier": corr_mult, "aimc_multiplier": aimc_mult,
        })

    def get_multipliers(self, model_id: str) -> dict:
        """GET /api/v1/models/{model_id}/multipliers"""
        return self._request("GET", f"/api/v1/models/{model_id}/multipliers")

    # -- API keys ---------------------------------------------------------

    def get_api_keys(self) -> dict:
        """GET /api/v1/agents/keys — list API keys for current agent."""
        return self._request("GET", "/api/v1/agents/keys")

    def create_api_key(self, name: str = "additional") -> dict:
        """POST /api/v1/agents/keys — generate additional API key."""
        return self._request("POST", "/api/v1/agents/keys", params={"name": name})

    def revoke_api_key(self, key_id: str) -> dict:
        """DELETE /api/v1/agents/keys/{key_id}"""
        return self._request("DELETE", f"/api/v1/agents/keys/{key_id}")

    # -- staking ----------------------------------------------------------

    def stake(self, model_id: str, amount_usdc: float, wallet_address: str) -> dict:
        """Stake USDC on a model."""
        return self._request("POST", "/api/v1/staking/stake", json={
            "model_id": model_id,
            "amount_usdc": amount_usdc,
            "wallet_address": wallet_address,
        })

    def confirm_stake(self, stake_id: str, txn_hash: str) -> dict:
        """Confirm a stake with on-chain transaction hash."""
        return self._request("POST", "/api/v1/staking/confirm", json={
            "stake_id": stake_id, "txn_hash": txn_hash,
        })

    def unstake(self, stake_id: str) -> dict:
        """Unstake USDC from a model."""
        return self._request("POST", f"/api/v1/staking/unstake/{stake_id}")

    def get_stake_balance(self, model_id: str) -> dict:
        """Get current stake balance and pending payouts."""
        return self._request("GET", f"/api/v1/staking/balance/{model_id}")

    def claim_payout(self, model_id: str, round_id: str) -> dict:
        """Claim resolved payout for a model and round."""
        return self._request("POST", "/api/v1/staking/claim", json={
            "model_id": model_id, "round_id": round_id,
        })

    def get_staking_history(self, model_id: str) -> dict:
        """Get stake/unstake/claim history."""
        return self._request("GET", f"/api/v1/staking/history/{model_id}")

    # -- CREATE2 staking --------------------------------------------------

    def get_deposit_address(self) -> dict:
        """GET /api/v1/staking/deposit-address — CREATE2 deposit address."""
        return self._request("GET", "/api/v1/staking/deposit-address")

    def relay_stake(self, model_id: str) -> dict:
        """Forward deposited USDC to tournament staking contract."""
        return self._request("POST", "/api/v1/staking/relay-stake", json={"model_id": model_id})

    def relay_claim(self, round_id: int) -> dict:
        """Claim payout from a resolved round."""
        return self._request("POST", "/api/v1/staking/relay-claim", json={"round_id": round_id})

    def withdraw_usdc(self, to_address: str, amount_usdc: float) -> dict:
        """Withdraw USDC to external wallet."""
        return self._request("POST", "/api/v1/staking/withdraw", json={
            "to_address": to_address, "amount_usdc": amount_usdc,
        })

    def get_forwarder_balance(self) -> dict:
        """GET /api/v1/staking/forwarder-balance"""
        return self._request("GET", "/api/v1/staking/forwarder-balance")

    # -- seasons & benchmarks ---------------------------------------------

    def get_seasons(self) -> dict:
        """GET /api/v1/seasons — list tournament seasons."""
        return self._request("GET", "/api/v1/seasons")

    def get_current_season(self) -> dict:
        """GET /api/v1/seasons/current — current season with rankings."""
        return self._request("GET", "/api/v1/seasons/current")

    def get_benchmarks(self) -> dict:
        """GET /api/v1/benchmarks — benchmark models with performance stats."""
        return self._request("GET", "/api/v1/benchmarks")

    # -- notifications & badges -------------------------------------------

    def get_notifications(self, unread_only: bool = True, limit: int = 50) -> dict:
        """Get notifications (badge awards, altitude changes, round scores)."""
        params: dict[str, Any] = {"limit": limit}
        if unread_only:
            params["unread_only"] = "true"
        return self._request("GET", "/api/v1/notifications", params=params)

    def get_badges(self, model_id: str | None = None) -> dict:
        """Get all achievement badges earned."""
        if model_id:
            return self._request("GET", f"/api/v1/models/{model_id}/badges")
        return self._request("GET", "/api/v1/badges")

    def mark_notification_read(self, notification_id: str) -> dict:
        """Mark a notification as read."""
        return self._request("POST", f"/api/v1/notifications/{notification_id}/read")

    def mark_all_notifications_read(self) -> dict:
        """Mark all notifications as read."""
        return self._request("POST", "/api/v1/notifications/read-all")

    # -- local evaluation -------------------------------------------------

    @staticmethod
    def evaluate(
        predictions: "np.ndarray | pd.Series",
        validation_data: "pd.DataFrame",
        target: str = "target_everest_20",
    ) -> dict:
        """Evaluate predictions against validation set locally."""
        import numpy as np
        from scipy.stats import spearmanr, rankdata

        preds = np.asarray(predictions)
        targets = validation_data[target].values
        valid = ~np.isnan(targets)
        preds = preds[valid]
        targets = targets[valid]

        simple_c, _ = spearmanr(preds, targets)
        result: dict = {"simple_corr": round(float(simple_c), 6)}

        if "climb_difficulty" in validation_data.columns:
            weights = (validation_data["climb_difficulty"].values[valid] / 5.0).astype(np.float32)
            pred_ranks = rankdata(preds, method="average")
            target_ranks = rankdata(targets, method="average")
            w_sum = weights.sum()
            if w_sum > 0:
                pm = np.dot(weights, pred_ranks) / w_sum
                tm = np.dot(weights, target_ranks) / w_sum
                pd_dev = pred_ranks - pm
                td_dev = target_ranks - tm
                cov = np.dot(weights, pd_dev * td_dev) / w_sum
                pvar = np.dot(weights, pd_dev ** 2) / w_sum
                tvar = np.dot(weights, td_dev ** 2) / w_sum
                denom = np.sqrt(pvar * tvar)
                result["weighted_corr"] = round(float(cov / denom), 6) if denom > 0 else 0.0
            else:
                result["weighted_corr"] = 0.0

            difficulties = validation_data["climb_difficulty"].values[valid]
            breakdown = {}
            for d in sorted(set(int(x) for x in difficulties if not np.isnan(x))):
                d_mask = difficulties == d
                if d_mask.sum() >= 3:
                    c, _ = spearmanr(preds[d_mask], targets[d_mask])
                    breakdown[d] = round(float(c), 4) if not np.isnan(c) else 0.0
            result["per_difficulty"] = breakdown

        return result

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> EverestAPI:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
