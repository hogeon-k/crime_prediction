from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from pandas import DataFrame


class ProcessStatus(Enum):
    IDLE = auto()  # 대기
    RUNNING = auto()  # 쓰레드 실행 중
    SUCCESS = auto()  # 전체 성공
    FAILED = auto()  # Fail-Fast 중단


@dataclass
class ExcelResult:
    """단일 처리 단계 결과"""

    success: bool
    message: str
    data: DataFrame | None = field(default=None, repr=False)


@dataclass
class ExcelState:
    """ViewModel → View 로 전달되는 Observable 상태"""

    status: ProcessStatus = ProcessStatus.IDLE
    current_step: str = ""
    completed_steps: list[str] = field(default_factory=list)
    failed_step: str = ""
    error_message: str = ""
    final_data: DataFrame | None = field(default=None, repr=False)

    @property
    def is_running(self) -> bool:
        return self.status == ProcessStatus.RUNNING


@dataclass
class ColumnMapping:
    """원본 컬럼명 → 표준 컬럼명 매핑 테이블"""

    mapping: dict[str, str] = field(
        default_factory=lambda: {
            # 범죄 유형
            "범죄유형": "범죄_유형",
            "crime_type": "범죄_유형",
            "죄종": "범죄_유형",
            # 지역
            "지역명": "지역",
            "region": "지역",
            "시군구": "지역",
            # 날짜
            "날짜": "발생_일자",
            "date": "발생_일자",
            "발생일": "발생_일자",
            "발생년도": "연도",
            "year": "연도",
            # 발생 건수
            "건수": "발생_건수",
            "count": "발생_건수",
            "발생건수": "발생_건수",
            "crime_count": "발생_건수",
            # 인구수
            "인구": "인구수",
            "population": "인구수",
        }
    )


@dataclass
class ValidationRule:
    """검증 규칙"""

    required_columns: set[str] = field(
        default_factory=lambda: {"범죄_유형", "지역", "연도", "발생_건수"}
    )
    valid_years: range = field(default_factory=lambda: range(2015, 2027))
    allowed_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".xlsx", ".csv"})
    )
