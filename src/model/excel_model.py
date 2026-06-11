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
    # 현재 프로그램 상태를 저장
    status: ProcessStatus = ProcessStatus.IDLE
    current_step: str = ""
    completed_steps: list[str] = field(default_factory=list)
    failed_step: str = ""
    error_message: str = ""
    final_data: pd.DataFrame | None = None
    predicted_incidents: float | None = None
    predicted_rate: float | None = None


@dataclass
class UploadParams:
    """Excel/CSV 업로드 파라미터."""

    mode: str  # "standard" | "government"
    crime_files: list[str] = field(default_factory=list)
    pop_files: list[str] = field(default_factory=list)
    standard_file: str = ""

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.mode not in ("standard", "government"):
            errors.append("mode는 'standard' 또는 'government' 여야 합니다.")
            return errors

        if self.mode == "standard":
            if not self.standard_file.strip():
                errors.append("표준 양식 파일을 선택하세요.")
        else:
            if not self.crime_files:
                errors.append("범죄 데이터 파일을 1개 이상 선택하세요.")
            if not self.pop_files:
                errors.append("인구 데이터 파일을 1개 이상 선택하세요.")
        return errors


@dataclass(frozen=True)
class ValidationRule:

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

    valid_years: tuple[int, ...] = field(
        default_factory=lambda: tuple(range(2018, 2027))
    )
