from pathlib import Path
import pandas as pd

from model.excel_model import (
    ProcessResult,
    ValidationRule,
)


class CrimeService:

    def __init__(self):
        self._rule = ValidationRule()

    def load_and_merge(
        self,
        crime_files: list[str],
        pop_files: list[str],
    ) -> ProcessResult:

        try:

            crime_frames = []

            for file in crime_files:

                year = int(Path(file).stem[-4:])

                crime = pd.read_csv(
                    file,
                    encoding="utf-8-sig",
                )

                region_cols = [
                    col
                    for col in crime.columns
                    if col not in ["범죄대분류", "범죄중분류"]
                ]

                crime = crime.melt(
                    id_vars=["범죄대분류", "범죄중분류"],
                    value_vars=region_cols,
                    var_name="지역",
                    value_name="발생_건수",
                )

                crime["범죄_유형"] = crime["범죄중분류"]
                crime["연도"] = year

                crime_frames.append(
                    crime[
                        [
                            "범죄_유형",
                            "지역",
                            "연도",
                            "발생_건수",
                        ]
                    ]
                )

            crime_df = pd.concat(
                crime_frames,
                ignore_index=True,
            )

            pop_frames = []

            for file in pop_files:

                pop = pd.read_csv(
                    file,
                    encoding="utf-8-sig",
                )

                pop["지역"] = (
                    pop["시도명"].astype(str) + " " + pop["시군구명"].astype(str)
                )

                pop["연도"] = pop["기준연월"].astype(str).str[:4].astype(int)

                pop["인구수"] = pop["계"]

                pop = pop.groupby(
                    ["지역", "연도"],
                    as_index=False,
                )["인구수"].sum()

                pop_frames.append(pop)

            pop_df = pd.concat(
                pop_frames,
                ignore_index=True,
            )

            merged = pd.merge(
                crime_df,
                pop_df,
                on=["지역", "연도"],
                how="left",
            )

            return ProcessResult(
                True,
                "병합 성공",
                merged,
            )

        except Exception as exc:
            return ProcessResult(
                False,
                f"병합 실패: {exc}",
            )

    def validate(self, df: pd.DataFrame) -> ProcessResult:

        missing = self._rule.required_columns - set(df.columns)

        if missing:
            return ProcessResult(
                False,
                f"컬럼 누락: {missing}",
            )

        return ProcessResult(
            True,
            "검증 성공",
            df,
        )

    def handle_missing(
        self,
        df: pd.DataFrame,
    ) -> ProcessResult:

        df = df.copy()

        df.dropna(
            subset=["범죄_유형", "지역", "연도"],
            inplace=True,
        )

        df["발생_건수"] = df["발생_건수"].fillna(0)

        # 지역별 평균으로 채우기
        df["인구수"] = df.groupby("지역")["인구수"].transform(
            lambda x: x.fillna(x.mean())
        )

        return ProcessResult(
            True,
            "결측치 처리 완료",
            df,
        )

    def convert_types(
        self,
        df: pd.DataFrame,
    ) -> ProcessResult:

        try:

            df = df.copy()

            df["연도"] = pd.to_numeric(df["연도"]).astype(int)

            df["발생_건수"] = (
                pd.to_numeric(df["발생_건수"], errors="coerce").fillna(0).astype(int)
            )

            df["인구수"] = pd.to_numeric(df["인구수"]).astype(int)

            df["범죄율"] = (df["발생_건수"] / df["인구수"]) * 100000

            return ProcessResult(
                True,
                "타입 변환 성공",
                df,
            )

        except Exception as exc:
            return ProcessResult(
                False,
                f"변환 실패: {exc}",
            )
