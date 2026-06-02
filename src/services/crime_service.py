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
    "세종특별자치시": "세종시",  # 범죄 CSV: 구 없이 '세종시' 단독
    "경기도": "경기도",
    "강원특별자치도": "강원도",  # 범죄 CSV는 구 명칭 '강원도' 유지
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
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"지원하는 인코딩으로 파일을 읽을 수 없습니다: {file}")


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


def _extract_top_sigungu(name: str) -> str:
    """
    인구 CSV의 시군구명에서 상위 행정구역(시·군)만 추출합니다.

    인구 CSV는 특별시·광역시 산하 자치구와 대도시 산하 구까지
    세분화되어 있어 범죄 CSV의 '시' 단위와 불일치합니다.

    '수원시 장안구' → '수원시'   (대도시 내부 구 제거)
    '고양시 덕양구' → '고양시'
    '종로구'        → '종로구'   (광역시 자치구는 그대로 유지)
    """
    s = str(name).strip()
    m = re.match(r"^(\S+시|\S+군)\s+\S+구$", s)
    return m.group(1) if m else s


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
        범죄 CSV를 읽어 long-format으로 변환합니다.

        주요 처리:
        - '외국 미국' 등 해외 지역 제거 (인구 데이터 없음)
        - melt 직후 발생_건수 숫자 변환 (원본에 '-', 'N/A' 혼재 가능)
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

            # 추가: 해외 지역 제거 (인구 CSV에 대응 데이터 없음)
            crime = crime[~crime["지역"].str.startswith("외국")].copy()

            crime["범죄_유형"] = crime["범죄중분류"]
            crime["연도"] = year
            crime["지역"] = _normalize_region(crime["지역"])

            # 추가: melt 직후 즉시 숫자 변환
            crime["발생_건수"] = pd.to_numeric(crime["발생_건수"], errors="coerce")

            frames.append(crime[["범죄_유형", "지역", "연도", "발생_건수"]])

        return pd.concat(frames, ignore_index=True)

    def _load_population(self, pop_files: list[str]) -> pd.DataFrame:
        """
        인구 CSV를 읽어 범죄 CSV와 merge 가능한 지역·연도 단위로 집계합니다.

        핵심 수정 3가지:
        ① 시도명 전체명 → 약칭 변환  ('서울특별시' → '서울')
        ② 세종 단독 처리            ('세종특별자치시 세종시' → '세종시')
        ③ 대도시 구 → 시 단위 통합  ('수원시 장안구' → '수원시')
           → 범죄 CSV가 시 단위 집계이므로 인구도 동일하게 맞춤
        """
        frames: list[pd.DataFrame] = []

        for file in pop_files:
            raw = _read_csv_safe(file)

            sido_short = raw["시도명"].map(_SIDO_SHORT).fillna(raw["시도명"])
            is_sejong = raw["시도명"] == "세종특별자치시"
            top_sigungu = raw["시군구명"].astype(str).apply(_extract_top_sigungu)

            지역 = _normalize_region(sido_short + " " + top_sigungu)
            지역[is_sejong] = "세종시"

            연도 = pd.to_numeric(
                raw["기준연월"].astype(str).str[:4], errors="coerce"
            ).astype("Int64")

            인구수 = pd.to_numeric(raw["계"], errors="coerce")

            # 수정: 컬럼을 개별 추가하지 않고 pd.concat으로 한 번에 조립
            #    → PerformanceWarning(DataFrame highly fragmented) 제거
            pop = pd.concat(
                [지역.rename("지역"), 연도.rename("연도"), 인구수.rename("인구수")],
                axis=1,
            )

            pop = pop.groupby(["지역", "연도"], as_index=False)["인구수"].sum()
            frames.append(pop)

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # 2단계: 컬럼 검증 + 연도 범위 검증
    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame) -> ProcessResult:
        missing = self._rule.required_columns - set(df.columns)
        if missing:
            return ProcessResult(False, f"컬럼 누락: {missing}")

        # 수정: valid_years 실제 사용
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

            # 수정: apply 람다 → 벡터 연산 (성능 개선)
            df["범죄율"] = (
                df["발생_건수"] / df["인구수"].replace(0, pd.NA) * 100_000
            ).fillna(0.0)

            return ProcessResult(True, "타입 변환 성공", df)

        except Exception as exc:
            return ProcessResult(False, f"변환 실패: {exc}")
