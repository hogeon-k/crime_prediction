from __future__ import annotations

import pandas as pd

from constants import (
    COL_YEAR,
    COL_CRIME_RATE,
    COL_CRIME_TYPE,
    COL_INCIDENTS,
    COL_REGION,
    PREDICTED_INCIDENTS_COLUMN,
    PREDICTED_RATE_COLUMN,
)


class StatisticsService:
    """예측/업로드 결과 DataFrame에서 통계와 차트 데이터를 만든다."""

    @staticmethod
    def _value_column(df: pd.DataFrame) -> str | None:
        if PREDICTED_INCIDENTS_COLUMN in df.columns:
            return PREDICTED_INCIDENTS_COLUMN
        if COL_INCIDENTS in df.columns:
            return COL_INCIDENTS
        return None

    @staticmethod
    def _rate_column(df: pd.DataFrame) -> str | None:
        if PREDICTED_RATE_COLUMN in df.columns:
            return PREDICTED_RATE_COLUMN
        if COL_CRIME_RATE in df.columns:
            return COL_CRIME_RATE
        return None

    @staticmethod
    def prediction_summary(df: pd.DataFrame) -> dict[str, float | int | None]:
        value_column = StatisticsService._value_column(df)
        rate_column = StatisticsService._rate_column(df)

        values = pd.to_numeric(df[value_column], errors="coerce") if value_column else None
        rates = pd.to_numeric(df[rate_column], errors="coerce") if rate_column else None

        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "avg_predicted_incidents": float(values.mean()) if values is not None else None,
            "max_predicted_incidents": float(values.max()) if values is not None else None,
            "avg_predicted_rate": float(rates.mean()) if rates is not None else None,
            "region_count": int(df[COL_REGION].nunique()) if COL_REGION in df.columns else 0,
            "crime_type_count": int(df[COL_CRIME_TYPE].nunique())
            if COL_CRIME_TYPE in df.columns
            else 0,
            "year_min": int(pd.to_numeric(df[COL_YEAR], errors="coerce").min())
            if COL_YEAR in df.columns and pd.to_numeric(df[COL_YEAR], errors="coerce").notna().any()
            else None,
            "year_max": int(pd.to_numeric(df[COL_YEAR], errors="coerce").max())
            if COL_YEAR in df.columns and pd.to_numeric(df[COL_YEAR], errors="coerce").notna().any()
            else None,
        }

    @staticmethod
    def group_by_region(df: pd.DataFrame) -> list[dict[str, float | str]]:
        value_column = StatisticsService._value_column(df)
        if value_column is None or COL_REGION not in df.columns:
            return []

        grouped = (
            df.assign(_value=pd.to_numeric(df[value_column], errors="coerce").fillna(0.0))
            .groupby(COL_REGION, as_index=False)["_value"]
            .sum()
            .sort_values("_value", ascending=False)
        )
        return [
            {"label": str(row[COL_REGION]), "value": float(row["_value"])}
            for _, row in grouped.iterrows()
        ]

    @staticmethod
    def group_by_crime_type(df: pd.DataFrame) -> list[dict[str, float | str]]:
        value_column = StatisticsService._value_column(df)
        if value_column is None or COL_CRIME_TYPE not in df.columns:
            return []

        grouped = (
            df.assign(_value=pd.to_numeric(df[value_column], errors="coerce").fillna(0.0))
            .groupby(COL_CRIME_TYPE, as_index=False)["_value"]
            .sum()
            .sort_values("_value", ascending=False)
        )
        return [
            {"label": str(row[COL_CRIME_TYPE]), "value": float(row["_value"])}
            for _, row in grouped.iterrows()
        ]

    @staticmethod
    def top_region_by_crime_rate(df: pd.DataFrame, n: int = 10) -> list[dict[str, float | str]]:
        rate_column = StatisticsService._rate_column(df)
        if rate_column is None or COL_REGION not in df.columns:
            return []

        grouped = (
            df.assign(_rate=pd.to_numeric(df[rate_column], errors="coerce").fillna(0.0))
            .groupby(COL_REGION, as_index=False)["_rate"]
            .mean()
            .sort_values("_rate", ascending=False)
            .head(n)
        )
        return [
            {"label": str(row[COL_REGION]), "value": float(row["_rate"])}
            for _, row in grouped.iterrows()
        ]
