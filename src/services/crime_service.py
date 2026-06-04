from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from model.excel_model import ProcessResult, ValidationRule

_ENCODINGS = ("utf-8-sig", "euc-kr", "cp949")

# 인구 CSV 시도명(전체) → 범죄 CSV 시도명(약칭) 매핑
#    예: '서울특별시' → '서울', '강원특별자치도' → '강원도'
_SIDO_SHORT: dict[str, str] = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종시",
    "경기도": "경기도",
    "강원특별자치도": "강원도",
    "충청북도": "충북",
    "충청남도": "충남",
    "전북특별자치도": "전북",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
}


def _read_csv_safe(file: str) -> pd.DataFrame:
    """인코딩을 순차적으로 시도하며 CSV를 읽어 반환합니다."""
    last_error: Exception = ValueError("알 수 없는 오류")
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue
    raise ValueError(
        f"지원하는 인코딩으로 파일을 읽을 수 없습니다: {file}"
    ) from last_error


def _extract_year(filename: str) -> int:
    """파일명에서 연도(20XX)를 정규식으로 추출합니다."""
    match = re.search(r"(20\d{2})", Path(filename).stem)
    if match is None:
        raise ValueError(
            f"파일명에서 연도를 추출할 수 없습니다: {filename}\n"
            "파일명에 'YYYY' 형식의 연도(예: crime_2023.csv)가 포함되어야 합니다."
        )
    return int(match.group(1))


def _normalize_region(series: pd.Series) -> pd.Series:
    """지역명 정규화: 앞뒤 공백 제거, 연속 공백 단일화."""
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
        """
        범죄 CSV를 읽어 시도 단위 long-format으로 변환합니다.

        연도별 집계 단위 차이 처리:
        - 2022·2023: 시도 단위 컬럼 ('서울', '부산', ...)
        - 2024     : 시군구 단위 컬럼 ('서울 종로구', '서울 중구', ...)
                     → 시도 단위로 합산하여 연도 간 통일
        """
        frames: list[pd.DataFrame] = []

        for file in crime_files:
            year = _extract_year(file)
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

            crime = crime[~crime["지역"].str.startswith("외국")].copy()
            crime["범죄_유형"] = crime["범죄중분류"]
            crime["연도"] = year
            crime["지역"] = _normalize_region(crime["지역"])
            crime["발생_건수"] = pd.to_numeric(crime["발생_건수"], errors="coerce")

            # 시군구 단위(공백 포함) → 시도만 추출 후 합산
            if " " in region_cols[0]:
                crime["지역"] = crime["지역"].str.split(" ").str[0]
                crime = pd.DataFrame(
                    crime.groupby(["범죄_유형", "지역", "연도"], as_index=False)[
                        "발생_건수"
                    ].sum()
                )

            frames.append(crime[["범죄_유형", "지역", "연도", "발생_건수"]])

        return pd.concat(frames, ignore_index=True)

    def _load_population(self, pop_files: list[str]) -> pd.DataFrame:
        """
        인구 CSV를 읽어 시도·연도 단위로 집계합니다.

        연도별 컬럼명 차이 처리:
        - 2022·2023: '통계년월'
        - 2024     : '기준연월'
        """
        frames: list[pd.DataFrame] = []

        for file in pop_files:
            raw = _read_csv_safe(file)

            sido_short = raw["시도명"].map(_SIDO_SHORT).fillna(raw["시도명"])

            연월_col = "기준연월" if "기준연월" in raw.columns else "통계년월"
            연도 = pd.to_numeric(
                raw[연월_col].astype(str).str[:4], errors="coerce"
            ).astype("Int64")

            인구수 = pd.to_numeric(raw["계"], errors="coerce")

            pop = pd.concat(
                [
                    sido_short.rename("지역"),
                    연도.rename("연도"),
                    인구수.rename("인구수"),
                ],
                axis=1,
            )

            pop = pd.DataFrame(
                pop.groupby(["지역", "연도"], as_index=False)["인구수"].sum()
            )
            frames.append(pop)

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # 2단계: 컬럼 검증 + 연도 범위 검증
    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame) -> ProcessResult:
        missing = self._rule.required_columns - set(df.columns)
        if missing:
            return ProcessResult(False, f"컬럼 누락: {missing}")

        if "연도" in df.columns:
            valid_set = set(self._rule.valid_years)
            actual_years = set(df["연도"].dropna().unique())
            invalid_years = actual_years - valid_set
            if invalid_years:
                return ProcessResult(
                    False,
                    f"유효하지 않은 연도 값 포함: {sorted(invalid_years)}\n"
                    f"허용 범위: {min(valid_set)}~{max(valid_set)}",
                )

        return ProcessResult(True, "검증 성공", df)

    # ------------------------------------------------------------------
    # 3단계: 결측치 처리
    # ------------------------------------------------------------------
    def handle_missing(self, df: pd.DataFrame) -> ProcessResult:
        df = df.copy()

        df.dropna(subset=["범죄_유형", "지역", "연도"], inplace=True)
        df["발생_건수"] = df["발생_건수"].fillna(0)

        df["인구수"] = df.groupby("지역")["인구수"].transform(
            lambda x: x.fillna(x.mean())
        )
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
            df["인구수"] = (
                pd.to_numeric(df["인구수"], errors="coerce").fillna(0).astype(int)
            )

            df["범죄율"] = (
                df["발생_건수"] / df["인구수"].replace(0, pd.NA) * 100_000
            ).fillna(0.0)

            return ProcessResult(True, "타입 변환 성공", df)

        except Exception as exc:
            return ProcessResult(False, f"변환 실패: {exc}")
