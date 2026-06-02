from __future__ import annotations
from pathlib import Path
import pandas as pd

from model.excel_model import ExcelResult, ColumnMapping, ValidationRule


class ExcelService:
    """
    Excel 처리 순수 함수 집합 (Fail-Fast)
    상태 없음 — 입력을 받아 ExcelResult 를 반환할 뿐
    """

    def __init__(
        self,
        mapping: ColumnMapping | None = None,
        rule: ValidationRule | None = None,
    ):
        self._mapping = mapping or ColumnMapping()
        self._rule = rule or ValidationRule()

    # ── Step 1 : 파일 로드 ──────────────────────────────
    def load_excel(self, file_path: str) -> ExcelResult:
        path = Path(file_path)

        if path.suffix.lower() not in self._rule.allowed_extensions:
            return ExcelResult(
                success=False,
                message=f"확장자 오류: '{path.suffix}' 허용 확장자 → "
                f"{', '.join(sorted(self._rule.allowed_extensions))}",
            )

        try:
            df = (
                pd.read_csv(file_path, encoding="utf-8-sig")
                if path.suffix.lower() == ".csv"
                else pd.read_excel(file_path)
            )
            return ExcelResult(
                success=True, message=f"파일 로드 성공 ({len(df)}행)", data=df
            )
        except Exception as exc:
            return ExcelResult(success=False, message=f"파일 로드 실패: {exc}")

    # ── Step 2 : 컬럼명 통일 ────────────────────────────
    def unify_columns(self, df: pd.DataFrame) -> ExcelResult:
        """
        원본 컬럼명을 ColumnMapping 기준으로 표준화한 뒤
        필수 컬럼 누락 여부를 확인
        """
        df = df.copy()

        # 매핑 적용 (대소문자·공백 무시)
        rename = {}
        for col in df.columns:
            normalized = col.strip().replace(" ", "")
            if normalized in self._mapping.mapping:
                rename[col] = self._mapping.mapping[normalized]
        df.rename(columns=rename, inplace=True)

        # 필수 컬럼 누락 확인
        missing = self._rule.required_columns - set(df.columns)
        if missing:
            return ExcelResult(
                success=False,
                message=f"컬럼 누락: {', '.join(sorted(missing))} "
                f"(현재 컬럼: {list(df.columns)})",
            )

        renamed_info = f" ({len(rename)}개 컬럼 통일)" if rename else ""
        return ExcelResult(
            success=True,
            message=f"컬럼 통일 성공{renamed_info}",
            data=df,
        )

    # ── Step 3 : 결측치 처리 ────────────────────────────
    def handle_missing_values(self, df: pd.DataFrame) -> ExcelResult:
        try:
            df = df.copy()
            notes: list[str] = []

            # 범죄_유형 — 결측 행 제거
            if df["범죄_유형"].isna().any():
                cnt = df["범죄_유형"].isna().sum()
                df.dropna(subset=["범죄_유형"], inplace=True)
                notes.append(f"'범죄_유형' {cnt}행 제거")

            # 지역 — 결측 행 제거
            if df["지역"].isna().any():
                cnt = df["지역"].isna().sum()
                df.dropna(subset=["지역"], inplace=True)
                notes.append(f"'지역' {cnt}행 제거")

            # 연도 — 결측 행 제거 (기준 컬럼이므로 보정 불가)
            if df["연도"].isna().any():
                cnt = df["연도"].isna().sum()
                df.dropna(subset=["연도"], inplace=True)
                notes.append(f"'연도' {cnt}행 제거")

            # 발생_건수 — 0 대체
            if df["발생_건수"].isna().any():
                cnt = df["발생_건수"].isna().sum()
                df["발생_건수"] = df["발생_건수"].fillna(0)
                notes.append(f"'발생_건수' {cnt}개 → 0 대체")

            # 인구수 컬럼이 있을 때만 — 평균 대체
            if "인구수" in df.columns and df["인구수"].isna().any():
                mean_val = pd.to_numeric(df["인구수"], errors="coerce").mean()
                if pd.isna(mean_val):
                    return ExcelResult(
                        success=False,
                        message="결측치 보정 불가: '인구수' 전체가 결측치입니다.",
                    )
                cnt = df["인구수"].isna().sum()
                df["인구수"] = df["인구수"].fillna(round(mean_val))
                notes.append(f"'인구수' {cnt}개 → 평균({round(mean_val):,}) 대체")

            if df.empty:
                return ExcelResult(
                    success=False,
                    message="데이터 손상: 결측치 처리 후 데이터가 없습니다.",
                )

            msg = "결측치 처리 성공" + (
                ": " + " / ".join(notes) if notes else " (결측치 없음)"
            )
            return ExcelResult(success=True, message=msg, data=df)

        except Exception as exc:
            return ExcelResult(success=False, message=f"결측치 보정 불가: {exc}")

    # ── Step 4 : 타입 변환 ──────────────────────────────
    def convert_types(self, df: pd.DataFrame) -> ExcelResult:
        """
        범죄_유형 → str
        지역      → str
        연도      → int
        발생_건수 → int
        발생_일자 → datetime (컬럼이 있을 때만)
        인구수    → int     (컬럼이 있을 때만)
        """
        try:
            df = df.copy()

            df["범죄_유형"] = df["범죄_유형"].astype(str).str.strip()
            df["지역"] = df["지역"].astype(str).str.strip()
            df["연도"] = pd.to_numeric(df["연도"], errors="raise").astype(int)
            df["발생_건수"] = pd.to_numeric(df["발생_건수"], errors="raise").astype(int)

            if "발생_일자" in df.columns:
                df["발생_일자"] = pd.to_datetime(df["발생_일자"], errors="coerce")
                bad = df["발생_일자"].isna().sum()
                if bad > 0:
                    return ExcelResult(
                        success=False,
                        message=f"변환 실패: '발생_일자' {bad}개 변환 불가 (형식 확인 필요)",
                    )

            if "인구수" in df.columns:
                df["인구수"] = pd.to_numeric(df["인구수"], errors="raise").astype(int)

            return ExcelResult(success=True, message="타입 변환 성공", data=df)

        except Exception as exc:
            return ExcelResult(success=False, message=f"변환 실패: {exc}")
