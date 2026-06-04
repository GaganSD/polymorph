"""CSV→log-line corpus adapters for distillation staging."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import (
    alibaba_gpu,
    api_failures,
    cicd_failures,
    cloudtrail_flaws,
    distsys_synth,
    python_tracebacks,
    security_synth,
    servicenow_itsm,
    syslog_cremev2,
    win_events,
)

if TYPE_CHECKING:
    from .stage import main, stage_all

__all__ = [
    "alibaba_gpu",
    "api_failures",
    "cicd_failures",
    "cloudtrail_flaws",
    "distsys_synth",
    "python_tracebacks",
    "security_synth",
    "servicenow_itsm",
    "syslog_cremev2",
    "win_events",
    "main",
    "stage_all",
]


def __getattr__(name: str):
    if name in {"main", "stage_all"}:
        from .stage import main, stage_all

        return main if name == "main" else stage_all
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
