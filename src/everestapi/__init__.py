"""EverestAPI — Python SDK for the EverestQuant prediction tournament platform."""

__version__ = "0.1.1"

from everestapi.client import EverestAPI, EverestError
from everestapi.types import (
    FeatureResponse,
    Instrument,
    LeaderboardEntry,
    LeaderboardResponse,
    Score,
    ScoresResponse,
    StakeBalance,
    StakeResponse,
    Submission,
    UniverseResponse,
)

__all__ = [
    "EverestAPI",
    "EverestError",
    "FeatureResponse",
    "Instrument",
    "LeaderboardEntry",
    "LeaderboardResponse",
    "Score",
    "ScoresResponse",
    "StakeBalance",
    "StakeResponse",
    "Submission",
    "UniverseResponse",
]
