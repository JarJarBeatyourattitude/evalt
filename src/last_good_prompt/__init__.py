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

__all__ = [
    "BudgetExceeded",
    "Client",
    "DraftAnswer",
    "Example",
    "ModelResult",
    "OptimizationResult",
    "ProviderError",
    "Turn",
]

__version__ = "0.8.22"
