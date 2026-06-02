from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from model.excel_model import (
    ProcessResult,
    ValidationRule,
)

# ✅ 추가: 지원 인코딩 목록 (공공데이터는 utf-8-sig / euc-kr / cp949 혼재)
_ENCODINGS = ("utf-8-sig", "euc-kr", "cp949")


def _read_csv_safe(file: str) -> pd.DataFrame:
    """인코딩을 순차적으로 시도하며 CSV를 읽어 반환합니다."""
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"지원하는 인코딩으로 파일을 읽을 수 없습니다: {file}")


def _extract_year(filename: str) -> int:
    """
    파일명에서 연도(20XX)를 정규식으로 추출합니다.
    ✅ 수정: 기존 [-4:] 슬라이싱은 파일명 형식에 따라 ValueError 발생 가능
    """
    match = re.search(r"(20\d{2})", Path(filename).stem)
    if match is None:
        raise ValueError(
            f"파일명에서 연도를 추출할 수 없습니다: {filename}\n"
            "파일명에 'YYYY' 형식의 연도(예: crime_2023.csv)가 포함되어야 합니다."
        )
    return int(match.group(1))


def _normalize_region(series: pd.Series) -> pd.Series:
    """
    ✅ 추가: 지역명 정규화 (앞뒤 공백 제거, 연속 공백 단일화)
    범죄 CSV와 인구 CSV의 지역명 형식이 달라 merge 후 NaN이 발생하는 것을 방지
    """
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


class CrimeService:

    def __init__(self) -> None:
        self._rule = ValidationRule()

    # ------------------------------------------------------------------
    # 1단계: 파일 로드 & 병합
    # ------------------------------------------------------------------
    def load_and_merge(
        self,
        crime_files: list[str],
        pop_files: list[str],
    ) -> ProcessResult:

        try:
            crime_df = self._load_crime(crime_files)
            pop_df = self._load_population(pop_files)

            merged = pd.merge(
                crime_df,
                pop_df,
                on=["지역", "연도"],
                how="left",
            )

            return ProcessResult(True, "병합 성공", merged)

        except Exception as exc:
            return ProcessResult(False, f"병합 실패: {exc}")

    def _load_crime(self, crime_files: list[str]) -> pd.DataFrame:
        """범죄 CSV 파일들을 읽어 long-format으로 변환 후 합칩니다."""
        frames: list[pd.DataFrame] = []

        for file in crime_files:
            # ✅ 수정: 정규식 기반 연도 추출
            year = _extract_year(file)

            # ✅ 수정: 인코딩 자동 감지
            crime = _read_csv_safe(file)

            region_cols = [
                col for col in crime.columns if col not in ["범죄대분류", "범죄중분류"]
            ]

            crime = crime.melt(
                id_vars=["범죄대분류", "범죄중분류"],
                value_vars=region_cols,
                var_name="지역",
                value_name="발생_건수",
            )

            crime["범죄_유형"] = crime["범죄중분류"]
            crime["연도"] = year

            # ✅ 추가: 지역명 정규화
            crime["지역"] = _normalize_region(crime["지역"])

            frames.append(crime[["범죄_유형", "지역", "연도", "발생_건수"]])

        return pd.concat(frames, ignore_index=True)

    def _load_population(self, pop_files: list[str]) -> pd.DataFrame:
        """인구 CSV 파일들을 읽어 지역·연도 단위로 집계 후 합칩니다."""
        frames: list[pd.DataFrame] = []

        for file in pop_files:
            # ✅ 수정: 인코딩 자동 감지
            pop = _read_csv_safe(file)

            pop["지역"] = pop["시도명"].astype(str) + " " + pop["시군구명"].astype(str)

            # ✅ 추가: 지역명 정규화
            pop["지역"] = _normalize_region(pop["지역"])

            pop["연도"] = pop["기준연월"].astype(str).str[:4].astype(int)
            pop["인구수"] = pop["계"]

            pop = pop.groupby(["지역", "연도"], as_index=False)["인구수"].sum()
            frames.append(pop)

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # 2단계: 컬럼 검증
    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame) -> ProcessResult:
        missing = self._rule.required_columns - set(df.columns)

        if missing:
            return ProcessResult(False, f"컬럼 누락: {missing}")

        return ProcessResult(True, "검증 성공", df)

    # ------------------------------------------------------------------
    # 3단계: 결측치 처리
    # ------------------------------------------------------------------
    def handle_missing(self, df: pd.DataFrame) -> ProcessResult:
        df = df.copy()

        df.dropna(subset=["범죄_유형", "지역", "연도"], inplace=True)

        df["발생_건수"] = df["발생_건수"].fillna(0)

        # 지역별 평균으로 채우기
        df["인구수"] = df.groupby("지역")["인구수"].transform(
            lambda x: x.fillna(x.mean())
        )

        # ✅ 추가: 지역 전체가 NaN인 경우(평균도 NaN) → 전체 평균으로 2차 보정
        global_mean = df["인구수"].mean()
        df["인구수"] = df["인구수"].fillna(global_mean if pd.notna(global_mean) else 0)

        return ProcessResult(True, "결측치 처리 완료", df)

    # ------------------------------------------------------------------
    # 4단계: 타입 변환 및 범죄율 계산
    # ------------------------------------------------------------------
    def convert_types(self, df: pd.DataFrame) -> ProcessResult:
        try:
            df = df.copy()

            df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")

            df["발생_건수"] = (
                pd.to_numeric(df["발생_건수"], errors="coerce").fillna(0).astype(int)
            )

            # ✅ 수정: errors="coerce" + fillna(0) 추가 (NaN → int 변환 crash 방지)
            df["인구수"] = (
                pd.to_numeric(df["인구수"], errors="coerce").fillna(0).astype(int)
            )

            # ✅ 추가: 인구수 0 division 방지
            df["범죄율"] = df.apply(
                lambda row: (
                    (row["발생_건수"] / row["인구수"]) * 100_000
                    if row["인구수"] > 0
                    else 0.0
                ),
                axis=1,
            )

            return ProcessResult(True, "타입 변환 성공", df)

        except Exception as exc:
            return ProcessResult(False, f"변환 실패: {exc}")
