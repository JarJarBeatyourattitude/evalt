"""Public SDK surface for Evalt."""

from .core import (
    BudgetExceeded,
    Client,
    DraftAnswer,
    Evalt,
    Example,
    GateReport,
    ModelResult,
    OptimizationResult,
    ProviderError,
    RolePlan,
    RoutedAnswer,
    Suite,
    SuiteDraft,
    Turn,
    check_result,
    select_role_plan,
)
from .reporting import (
    compare_results,
    render_comparison_html,
    render_html_report,
    render_junit_report,
    write_reports,
)

__all__ = [
    "BudgetExceeded",
    "Client",
    "DraftAnswer",
    "Evalt",
    "Example",
    "GateReport",
    "ModelResult",
    "OptimizationResult",
    "ProviderError",
    "RolePlan",
    "RoutedAnswer",
    "Suite",
    "SuiteDraft",
    "Turn",
    "check_result",
    "select_role_plan",
    "compare_results",
    "render_comparison_html",
    "render_html_report",
    "render_junit_report",
    "write_reports",
]

__version__ = "0.8.22"
