"""Compatibility import for the pre-release Last Good Prompt package name."""

from .core import (
    BudgetExceeded,
    Client,
    DraftAnswer,
    Example,
    ModelResult,
    OptimizationResult,
    ProviderError,
    Turn,
)
from .scorers import (
    CommandScorer,
    CustomScorerError,
    ScoreRequest,
    ScoreResult,
    Scorer,
)

__all__ = [
    "BudgetExceeded",
    "Client",
    "DraftAnswer",
    "Example",
    "ModelResult",
    "OptimizationResult",
    "ProviderError",
    "Turn",
    "CommandScorer",
    "CustomScorerError",
    "ScoreRequest",
    "ScoreResult",
    "Scorer",
]

__version__ = "0.10.31"
