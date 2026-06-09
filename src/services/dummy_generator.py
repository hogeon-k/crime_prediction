from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# 지역별 인구 기준값 (mean, std) — 단위: 명
# ──────────────────────────────────────────────
_REGION_POP: Dict[str, tuple[float, float]] = {
    "서울": (9_500_000, 200_000),
    "부산": (3_400_000, 100_000),
    "인천": (3_000_000, 100_000),
    "대구": (2_400_000, 80_000),
    "대전": (1_460_000, 50_000),
    "광주": (1_450_000, 50_000),
    "울산": (1_100_000, 40_000),
    "수원": (1_200_000, 50_000),
    "고양": (1_050_000, 40_000),
    "용인": (1_080_000, 40_000),
    "창원": (980_000, 35_000),
    "성남": (920_000, 30_000),
    "청주": (850_000, 30_000),
    "전주": (650_000, 25_000),
    "천안": (660_000, 25_000),
    "안산": (650_000, 20_000),
    "남양주": (730_000, 25_000),
    "화성": (900_000, 40_000),
    "진주": (350_000, 15_000),
    "제주": (700_000, 25_000),
    "세종": (390_000, 30_000),
}
_DEFAULT_POP = (300_000, 50_000)  # 목록에 없는 지역 기본값

# ──────────────────────────────────────────────
# 범죄 유형별 포아송 λ값
# ──────────────────────────────────────────────
_CRIME_LAMBDA: Dict[str, float] = {
    "살인": 2.0,
    "강도": 5.0,
    "강간·강제추행": 18.0,
    "절도": 120.0,
    "폭력": 90.0,
    "사기": 200.0,
    "방화": 3.0,
    "마약": 15.0,
}

VALID_REGIONS: List[str] = sorted(_REGION_POP.keys())


# ──────────────────────────────────────────────
# GenerationParams
# ──────────────────────────────────────────────
@dataclass
class GenerationParams:
    data_count: int
    year_start: int
    year_end: int
    region: List[str]

    def validate(self) -> bool:
        if not (1 <= self.data_count <= 10_000):
            return False
        if not (2020 <= self.year_start <= 2026):
            return False
        if not (self.year_start <= self.year_end <= 2026):
            return False
        if len(self.region) < 1:
            return False
        return True

    def validation_errors(self) -> List[str]:
        errors: List[str] = []
        if not (1 <= self.data_count <= 10_000):
            errors.append("데이터 수는 1 이상 10,000 이하여야 합니다.")
        if not (2020 <= self.year_start <= 2026):
            errors.append("시작 연도는 2020~2026 사이여야 합니다.")
        if not (self.year_start <= self.year_end <= 2026):
            errors.append("종료 연도는 시작 연도 이상 2026 이하여야 합니다.")
        if len(self.region) < 1:
            errors.append("지역을 1개 이상 선택해야 합니다.")
        return errors


@dataclass(frozen=True)
class ExportResult:
    success: bool
    message: str

    def __bool__(self) -> bool:
        return self.success


def run_generation_pipeline(
    gui_input: Dict[str, Any],
    seed: int | None = 42,
    on_state_update: Any = None,
) -> pd.DataFrame:
    """
    GUI에서 받은 설정 dict → 생성 → 후처리 파이프라인까지 한 번에 실행.

    gui_input 예시 (main_window._on_generate 와 동일):
        {
            "data_count": 500,
            "year_start": 2022,
            "year_end": 2024,
            "region": ["서울", "부산", "인천"],
        }

    Returns:
        후처리 완료된 DataFrame. 실패 시 ValueError 발생.
    """
    from model.excel_model import CrimeState, ProcessStatus
    from services.crime_service import CrimeService

    params = get_user_input_gui(gui_input)
    errors = params.validation_errors()
    if errors:
        raise ValueError("\n".join(errors))

    gen = DataGenerator(params, seed=seed)
    df_raw = gen.generate()

    callback = on_state_update if on_state_update is not None else (lambda _s: None)
    service = CrimeService()
    state = CrimeState(status=ProcessStatus.RUNNING)
    df = df_raw

    for name, fn in [
        ("검증", service.validate),
        ("결측치 처리", service.handle_missing),
        ("타입 변환", service.convert_types),
    ]:
        state.current_step = name
        callback(state)
        result = fn(df)
        if not result.success:
            state.status = ProcessStatus.FAILED
            state.failed_step = name
            state.error_message = result.message
            callback(state)
            raise RuntimeError(f"파이프라인 실패 [{name}]: {result.message}")
        df = result.data
        state.completed_steps.append(name)

    state.status = ProcessStatus.SUCCESS
    state.final_data = df
    callback(state)
    return df


def get_user_input_gui(data: Dict[str, Any]) -> GenerationParams:
    """GUI dict → GenerationParams 변환. 타입 자동 변환 + 필수 키 검증."""
    required = {"data_count", "year_start", "year_end", "region"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"필수 키가 누락되었습니다: {missing}")

    data_count = int(data["data_count"])
    year_start = int(data["year_start"])
    year_end = int(data["year_end"])
    region = list(data["region"])

    if data_count < 1:
        raise ValueError("data_count는 1 이상이어야 합니다.")
    if not (2020 <= year_start <= 2026):
        raise ValueError("year_start는 2020~2026 사이여야 합니다.")
    if not (year_start <= year_end <= 2026):
        raise ValueError("year_end는 year_start 이상 2026 이하여야 합니다.")
    if len(region) < 1:
        raise ValueError("region은 1개 이상이어야 합니다.")

    return GenerationParams(
        data_count=data_count,
        year_start=year_start,
        year_end=year_end,
        region=region,
    )


# ──────────────────────────────────────────────
# DataGenerator
# ──────────────────────────────────────────────
class DataGenerator:
    def __init__(self, params: GenerationParams, seed: int | None = None) -> None:
        self._p = params
        self._rng = np.random.default_rng(seed)

    # ── 전체 생성 ──────────────────────────────
    def generate(self) -> pd.DataFrame:
        n = self._p.data_count
        regions = self.generate_regions()
        years = self.generate_years()
        crime_types = self.generate_crime_types()
        incidents = self._generate_incidents_per_type(crime_types)
        population = self.generate_population(regions)

        df = pd.DataFrame(
            {
                "범죄_유형": crime_types,
                "지역": regions,
                "연도": years,
                "발생_건수": incidents,
                "인구수": population,
            }
        )
        assert len(df) == n, "행 수 불일치"
        assert (df["발생_건수"] >= 0).all(), "발생_건수에 음수 포함"
        assert (df["인구수"] > 0).all(), "인구수에 0 이하 포함"
        return df

    # ── 지역 ──────────────────────────────────
    def generate_regions(self) -> List[str]:
        return list(
            self._rng.choice(self._p.region, size=self._p.data_count, replace=True)
        )

    # ── 연도 ──────────────────────────────────
    def generate_years(self) -> List[int]:
        years = list(range(self._p.year_start, self._p.year_end + 1))
        return [
            int(y)
            for y in self._rng.choice(years, size=self._p.data_count, replace=True)
        ]

    # ── 인구수 ────────────────────────────────
    def generate_population(
        self,
        regions: List[str] | None = None,
        mean: float | None = None,
        std: float | None = None,
    ) -> List[int]:
        if regions is None:
            regions = self.generate_regions()

        result: List[int] = []
        for r in regions:
            m, s = (
                _REGION_POP.get(r, _DEFAULT_POP)
                if (mean is None or std is None)
                else (mean, std)
            )
            val = int(max(1, self._rng.normal(m, s)))
            result.append(val)
        return result

    # ── 범죄 유형 ─────────────────────────────
    def generate_crime_types(self) -> List[str]:
        types = list(_CRIME_LAMBDA.keys())
        return list(self._rng.choice(types, size=self._p.data_count, replace=True))

    # ── 발생 건수 (포아송) ────────────────────
    def generate_incidents(self, lambda_param: float = 50.0) -> List[int]:
        return [
            int(x) for x in self._rng.poisson(lambda_param, size=self._p.data_count)
        ]

    def _generate_incidents_per_type(self, crime_types: List[str]) -> List[int]:
        return [
            int(self._rng.poisson(_CRIME_LAMBDA.get(ct, 50.0))) for ct in crime_types
        ]


# ──────────────────────────────────────────────
# DataExporter
# ──────────────────────────────────────────────
class DataExporter:

    @staticmethod
    def save_to_csv(
        df: pd.DataFrame,
        filepath: str,
        encoding: str = "utf-8-sig",
    ) -> ExportResult:
        try:
            df.to_csv(filepath, index=False, encoding=encoding)
            return ExportResult(True, f"CSV 저장 완료: {filepath}")
        except Exception as exc:
            return ExportResult(False, f"CSV 저장 실패: {exc}")

    @staticmethod
    def preview(df: pd.DataFrame, rows: int = 5) -> Dict[str, Any]:
        rows = max(5, rows)
        return {
            "data": df.head(rows).to_dict(orient="records"),
            "rows": len(df),
            "columns": len(df.columns),
            "column_list": list(df.columns),
        }
