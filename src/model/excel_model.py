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
    # ✅ 수정: set → frozenset (frozen dataclass는 unhashable 필드를 가지면 hash() 시 TypeError 발생)
    required_columns: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "범죄_유형",
                "지역",
                "연도",
                "발생_건수",
                "인구수",
            }
        )
    )
    # ✅ 수정: tuple 사용 (range는 hashable이지만 명시적 불변 타입으로 통일)
    valid_years: tuple[int, ...] = field(
        default_factory=lambda: tuple(range(2022, 2031))
    )
