from __future__ import annotations

from pathlib import Path

import pandas as pd

from constants import (
    COL_CRIME_RATE,
    COL_INCIDENTS,
    COL_YEAR,
    PREDICTED_INCIDENTS_COLUMN,
    PREDICTED_RATE_COLUMN,
)

DATA_KIND_COLUMN = "데이터_구분"
ACTUAL_KIND = "actual"
PREDICTED_KIND = "predicted"


class AnalysisDataService:
    """분석 화면에서 사용할 실제/예측 데이터를 로드하고 통합한다."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        actual_data_path: str | Path | None = None,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[2]
        self.actual_data_path = (
            Path(actual_data_path)
            if actual_data_path is not None
            else self.root_dir / "src" / "data" / "processed_crime_data.csv"
        )
        self.last_messages: list[str] = []

    def prediction_result_path(self, year: int) -> Path:
        return self.root_dir / "data" / f"prediction_result_{int(year)}.xlsx"

    def get_available_prediction_years(self, years: list[int] | None = None) -> list[int]:
        years = years or [2025, 2026, 2027]
        return [year for year in years if self.prediction_result_path(year).exists()]

    def load_actual_analysis_data(self) -> pd.DataFrame:
        if not self.actual_data_path.exists():
            raise FileNotFoundError(f"실제 데이터 파일을 찾을 수 없습니다: {self.actual_data_path}")
        df = self._read_file(self.actual_data_path)
        return self._normalize_for_analysis(df, ACTUAL_KIND)

    def load_prediction_results_by_year(self, years: list[int]) -> pd.DataFrame:
        frames = []
        self.last_messages = []

        for year in years:
            path = self.prediction_result_path(year)
            if not path.exists():
                self.last_messages.append(f"{year} 예측 결과 파일이 없습니다.")
                continue
            df = self._read_file(path)
            frames.append(self._normalize_for_analysis(df, PREDICTED_KIND, year=year))

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def load_combined_actual_prediction_data(
        self,
        prediction_years: list[int] | None = None,
    ) -> pd.DataFrame:
        prediction_years = prediction_years or [2025, 2026, 2027]
        actual_df = self.load_actual_analysis_data()
        predicted_df = self.load_prediction_results_by_year(prediction_years)

        if predicted_df.empty:
            return actual_df
        return pd.concat([actual_df, predicted_df], ignore_index=True, sort=False)

    def get_yearly_crime_rate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or COL_YEAR not in df.columns or COL_CRIME_RATE not in df.columns:
            return pd.DataFrame(columns=[COL_YEAR, DATA_KIND_COLUMN, COL_CRIME_RATE])

        working = df.copy()
        if DATA_KIND_COLUMN not in working.columns:
            working[DATA_KIND_COLUMN] = ACTUAL_KIND

        working["_year"] = pd.to_numeric(working[COL_YEAR], errors="coerce")
        working["_rate"] = pd.to_numeric(working[COL_CRIME_RATE], errors="coerce")
        summary = (
            working.dropna(subset=["_year", "_rate"])
            .groupby(["_year", DATA_KIND_COLUMN], as_index=False)["_rate"]
            .mean()
            .rename(columns={"_year": COL_YEAR, "_rate": COL_CRIME_RATE})
            .sort_values([COL_YEAR, DATA_KIND_COLUMN])
        )
        summary[COL_YEAR] = summary[COL_YEAR].astype(int)
        return summary

    def _read_file(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        raise ValueError("지원하지 않는 파일 형식입니다. csv, xlsx, xls 파일만 사용할 수 있습니다.")

    def _normalize_for_analysis(
        self,
        df: pd.DataFrame,
        data_kind: str,
        year: int | None = None,
    ) -> pd.DataFrame:
        normalized = df.copy()
        if year is not None:
            normalized[COL_YEAR] = int(year)
        if PREDICTED_RATE_COLUMN in normalized.columns:
            normalized[COL_CRIME_RATE] = normalized[PREDICTED_RATE_COLUMN]
        if PREDICTED_INCIDENTS_COLUMN in normalized.columns:
            normalized[COL_INCIDENTS] = normalized[PREDICTED_INCIDENTS_COLUMN]
        normalized[DATA_KIND_COLUMN] = data_kind
        return normalized
