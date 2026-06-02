from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import pandas as pd


class ProcessStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()


@dataclass
class ProcessResult:
    success: bool
    message: str
    data: pd.DataFrame | None = None


@dataclass
class CrimeState:
    status: ProcessStatus = ProcessStatus.IDLE

    current_step: str = ""

    completed_steps: list[str] = field(default_factory=list)

    failed_step: str = ""

    error_message: str = ""

    final_data: pd.DataFrame | None = None


@dataclass(frozen=True)
class ValidationRule:

    required_columns: set[str] = field(
        default_factory=lambda: {
            "범죄_유형",
            "지역",
            "연도",
            "발생_건수",
            "인구수",
        }
    )

    valid_years: range = field(default_factory=lambda: range(2022, 2031))
