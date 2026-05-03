"""EverestAPI — Response types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Instrument:
    """A single instrument in the tournament universe."""

    ticker: str
    name: str
    asset_class: str
    sector: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Instrument:
        return cls(
            ticker=d["ticker"],
            name=d["name"],
            asset_class=d["asset_class"],
            sector=d.get("sector", ""),
        )


@dataclass
class UniverseResponse:
    """Response from GET /api/v1/universe."""

    date: str
    instruments: list[Instrument]
    count: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UniverseResponse:
        return cls(
            date=d["date"],
            instruments=[Instrument.from_dict(i) for i in d["instruments"]],
            count=d["count"],
        )


@dataclass
class FeatureResponse:
    """Response from GET /api/v1/features."""

    date: str
    features: dict[str, dict[str, float]]
    feature_names: list[str]
    count: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureResponse:
        return cls(
            date=d["date"],
            features=d["features"],
            feature_names=d["feature_names"],
            count=d["count"],
        )


@dataclass
class Submission:
    """Response from POST /api/v1/predictions."""

    submission_id: str
    model_id: str
    date: str
    instrument_count: int
    submitted_at: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Submission:
        return cls(
            submission_id=d["submission_id"],
            model_id=d["model_id"],
            date=d["date"],
            instrument_count=d["instrument_count"],
            submitted_at=d["submitted_at"],
        )


@dataclass
class Score:
    """A single day's scoring result."""

    date: str
    corr: float
    aimc: float
    payout: float
    cumulative_payout: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Score:
        return cls(
            date=d["date"],
            corr=d["corr"],
            aimc=d["aimc"],
            payout=d["payout"],
            cumulative_payout=d["cumulative_payout"],
        )


@dataclass
class ScoresResponse:
    """Response from GET /api/v1/scores."""

    model_id: str
    scores: list[Score]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoresResponse:
        return cls(
            model_id=d["model_id"],
            scores=[Score.from_dict(s) for s in d["scores"]],
        )


@dataclass
class LeaderboardEntry:
    """A single entry on the leaderboard."""

    model_id: str
    agent_name: str
    mean_corr: float
    mean_aimc: float
    sharpe: float
    total_payout: float
    rank: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LeaderboardEntry:
        return cls(
            model_id=d["model_id"],
            agent_name=d["agent_name"],
            mean_corr=d["mean_corr"],
            mean_aimc=d["mean_aimc"],
            sharpe=d["sharpe"],
            total_payout=d["total_payout"],
            rank=d["rank"],
        )


@dataclass
class LeaderboardResponse:
    """Response from GET /api/v1/leaderboard."""

    period: str
    entries: list[LeaderboardEntry]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LeaderboardResponse:
        return cls(
            period=d["period"],
            entries=[LeaderboardEntry.from_dict(e) for e in d["entries"]],
        )


@dataclass
class StakeResponse:
    """Response from POST /api/v1/staking/stake."""

    id: str
    model_id: str
    amount_usdc: float
    wallet_address: str
    status: str
    testnet: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StakeResponse:
        return cls(
            id=d["id"],
            model_id=d["model_id"],
            amount_usdc=d["amount_usdc"],
            wallet_address=d["wallet_address"],
            status=d["status"],
            testnet=d.get("testnet", True),
        )


@dataclass
class StakeBalance:
    """Response from GET /api/v1/staking/balance/{model_id}."""

    model_id: str
    total_staked: float
    pending_payouts: float
    total_earned: float
    testnet: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StakeBalance:
        return cls(
            model_id=d["model_id"],
            total_staked=d["total_staked"],
            pending_payouts=d["pending_payouts"],
            total_earned=d["total_earned"],
            testnet=d.get("testnet", True),
        )
