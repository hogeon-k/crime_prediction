from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from model.excel_model import (
    CrimeState,
    ProcessStatus,
    UploadParams,
)
from services.ai_service import AIService
from services.analysis_data_service import AnalysisDataService
from services.crime_service import CrimeService
from services.dummy_generator import DataExporter
from services.excel_pipeline import run_excel_pipeline
from services.statistics_service import StatisticsService


class CrimeViewModel:

    def __init__(
        self,
        callback: Callable[[CrimeState], None],
        service: CrimeService | None = None,
        ai_service: AIService | None = None,
        analysis_data_service: AnalysisDataService | None = None,
        statistics_service: StatisticsService | None = None,
    ) -> None:
        self._callback = callback
        self._service = service or CrimeService()
        self._ai_service = ai_service or AIService()
        self._analysis_data_service = analysis_data_service or AnalysisDataService()
        self._statistics_service = statistics_service or StatisticsService()
        self.state = CrimeState()
        self.selected_region: str | None = None
        self.selected_year: int | None = None
        self.analysis_messages: list[str] = []

    def process(
        self,
        crime_files: list[str],
        pop_files: list[str],
    ) -> None:
        """
        파이프라인 단계를 순서대로 실행합니다.
        각 단계 실패 시 즉시 중단하고 FAILED 상태로 콜백합니다.

        파이프라인 구조:
            1. 데이터 병합  (crime_files, pop_files → DataFrame)
            2. 검증         (DataFrame → DataFrame)
            3. 결측치 처리  (DataFrame → DataFrame)
            4. 타입 변환    (DataFrame → DataFrame)
        """

        steps = [
            (
                "데이터 병합",
                lambda _: self._service.load_and_merge(crime_files, pop_files),
            ),
            ("검증", self._service.validate),
            ("결측치 처리", self._service.handle_missing),
            ("타입 변환", self._service.convert_types),
        ]

        self.state = CrimeState(status=ProcessStatus.RUNNING)

        df = None

        for name, fn in steps:
            self.state.current_step = name
            self._callback(self.state)

            if name != "데이터 병합" and df is None:
                self._set_failed(name, "이전 단계 결과 데이터가 없습니다.")
                return

            result = fn(df)

            if not result.success:
                self._set_failed(name, result.message)
                return

            df = result.data
            self.state.completed_steps.append(name)

        self.state.status = ProcessStatus.SUCCESS
        self.state.final_data = df
        self._callback(self.state)

    def process_from_df(self, df: pd.DataFrame) -> None:
        """
        자동 생성된 DataFrame을 파이프라인에 투입합니다.
        병합 단계를 건너뛰고 검증 → 결측치 처리 → 타입 변환을 수행합니다.
        """
        steps = [
            ("검증", self._service.validate),
            ("결측치 처리", self._service.handle_missing),
            ("타입 변환", self._service.convert_types),
        ]

        self.state = CrimeState(status=ProcessStatus.RUNNING)

        for name, fn in steps:
            self.state.current_step = name
            self._callback(self.state)

            result = fn(df)

            if not result.success:
                self._set_failed(name, result.message)
                return

            df = result.data
            self.state.completed_steps.append(name)

        self.state.status = ProcessStatus.SUCCESS
        self.state.final_data = df
        self._callback(self.state)

    def process_upload(self, params: UploadParams) -> pd.DataFrame | None:
        """Excel/CSV 업로드 use case를 실행하고 결과를 state에 저장한다."""
        self.state = CrimeState(status=ProcessStatus.RUNNING)
        self._callback(self.state)

        try:
            df = run_excel_pipeline(params, on_state_update=self._replace_state)
        except Exception as exc:
            self._set_failed("업로드", str(exc))
            return None

        self.state.status = ProcessStatus.SUCCESS
        self.state.final_data = df
        self._callback(self.state)
        return df

    def predict_file(
        self,
        input_path: str,
        output_path: str,
        target_year: int,
    ) -> pd.DataFrame | None:
        """저장된 모델로 파일 예측을 실행하고 결과를 state에 저장한다."""
        self.state = CrimeState(status=ProcessStatus.RUNNING, current_step="파일 예측")
        self._callback(self.state)

        try:
            df = self._ai_service.predict_file(input_path, output_path, target_year=target_year)
        except Exception as exc:
            self._set_failed(
                "저장된 모델 예측",
                self._ai_service.format_prediction_error(str(exc)),
            )
            return None

        self.state.status = ProcessStatus.SUCCESS
        self.state.final_data = df
        self._callback(self.state)
        return df

    def predict_recursive_file(
        self,
        input_path: str,
        output_dir: str | Path,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame | None:
        """저장된 모델로 다년도 재귀 파일 예측을 실행하고 결과를 state에 저장한다."""
        self.state = CrimeState(status=ProcessStatus.RUNNING, current_step="다년도 재귀 예측")
        self._callback(self.state)

        try:
            results = self._ai_service.predict_recursive_file(
                input_path,
                output_dir,
                start_year=start_year,
                end_year=end_year,
            )
        except Exception as exc:
            self._set_failed(
                "다년도 재귀 예측",
                self._ai_service.format_prediction_error(str(exc)),
            )
            return None

        df = pd.concat(results.values(), ignore_index=True) if results else pd.DataFrame()
        self.state.status = ProcessStatus.SUCCESS
        self.state.final_data = df
        self._callback(self.state)
        return df

    def predict_one(
        self,
        year: int,
        region: str,
        crime_type: str,
        population: int,
        previous_incidents: float | None = None,
        previous_rate: float | None = None,
    ) -> float | None:
        """저장된 모델로 단건 예측을 실행한다."""
        self.state = CrimeState(status=ProcessStatus.RUNNING, current_step="단일 예측")
        self._callback(self.state)

        try:
            predicted_incidents = self._ai_service.predict_one(
                year=year,
                region=region,
                crime_type=crime_type,
                population=population,
                previous_incidents=previous_incidents,
                previous_rate=previous_rate,
            )
        except Exception as exc:
            self._set_failed(
                "단일 예측",
                self._ai_service.format_prediction_error(str(exc)),
            )
            return None

        self.state.status = ProcessStatus.SUCCESS
        self.state.predicted_incidents = float(predicted_incidents)
        self.state.predicted_rate = float(predicted_incidents / population * 100000)
        self._callback(self.state)
        return float(predicted_incidents)

    def save_dataframe(self, df: pd.DataFrame, path: str):
        return DataExporter.save_to_csv(df, path)

    def prediction_result_path(self, year: int) -> Path:
        return self._analysis_data_service.prediction_result_path(year)

    def load_dataframe_for_analysis(self, path: str | Path) -> pd.DataFrame | None:
        self.state = CrimeState(status=ProcessStatus.RUNNING, current_step="분석 데이터 로드")
        self._callback(self.state)

        try:
            input_path = Path(path)
            if not input_path.exists():
                raise FileNotFoundError(f"분석할 파일을 찾을 수 없습니다: {input_path}")
            if input_path.suffix.lower() == ".csv":
                df = pd.read_csv(input_path)
            elif input_path.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(input_path)
            else:
                raise ValueError("지원하지 않는 파일 형식입니다. csv, xlsx, xls 파일만 사용할 수 있습니다.")
        except Exception as exc:
            self._set_failed("분석 데이터 로드", str(exc))
            return None

        self.state.status = ProcessStatus.SUCCESS
        self.state.final_data = df
        self._callback(self.state)
        return df

    def load_actual_analysis_data(self) -> pd.DataFrame | None:
        return self._load_analysis_data("실제 데이터 로드", self._analysis_data_service.load_actual_analysis_data)

    def load_prediction_results_by_year(self, years: list[int]) -> pd.DataFrame | None:
        return self._load_analysis_data(
            "예측 결과 로드",
            lambda: self._analysis_data_service.load_prediction_results_by_year(years),
        )

    def load_combined_actual_prediction_data(self) -> pd.DataFrame | None:
        return self._load_analysis_data(
            "실제+예측 데이터 통합 로드",
            self._analysis_data_service.load_combined_actual_prediction_data,
        )

    def get_available_prediction_years(self) -> list[int]:
        return self._analysis_data_service.get_available_prediction_years()

    def get_yearly_crime_rate_summary(
        self,
        region_query: str = "",
        crime_type_query: str = "",
    ) -> pd.DataFrame:
        if self.state.final_data is None:
            return pd.DataFrame()
        return self._statistics_service.yearly_rate_summary(
            self.state.final_data,
            region=self.selected_region,
            region_query=region_query,
            crime_type_query=crime_type_query,
        )

    def set_region_year_selection(
        self,
        region: str | None = None,
        year: int | None = None,
    ) -> None:
        self.selected_region = region or None
        self.selected_year = year

    def get_prediction_summary(self) -> dict[str, float | int | None]:
        if self.state.final_data is None:
            return {}
        return self._statistics_service.prediction_summary(self.state.final_data)

    def get_region_chart_data(self) -> list[dict[str, float | str]]:
        if self.state.final_data is None:
            return []
        return self._statistics_service.group_by_region(self.state.final_data)

    def get_crime_type_chart_data(self) -> list[dict[str, float | str]]:
        if self.state.final_data is None:
            return []
        return self._statistics_service.group_by_crime_type(self.state.final_data)

    def get_top_rate_chart_data(self) -> list[dict[str, float | str]]:
        if self.state.final_data is None:
            return []
        return self._statistics_service.top_region_by_crime_rate(self.state.final_data)

    def get_available_regions(self) -> list[str]:
        if self.state.final_data is None:
            return []
        return self._statistics_service.available_regions(self.state.final_data)

    def get_available_years(self) -> list[int]:
        if self.state.final_data is None:
            return []
        return self._statistics_service.available_years(self.state.final_data)

    def get_filtered_records(
        self,
        region_query: str = "",
        crime_type_query: str = "",
        sort_by: str = "",
        ascending: bool = False,
    ) -> pd.DataFrame:
        if self.state.final_data is None:
            return pd.DataFrame()
        return self._statistics_service.filter_records(
            self.state.final_data,
            region=self.selected_region,
            year=self.selected_year,
            region_query=region_query,
            crime_type_query=crime_type_query,
            sort_by=sort_by,
            ascending=ascending,
        )

    def get_yearly_rate_trend(self) -> list[dict[str, float | int]]:
        if self.state.final_data is None:
            return []
        return self._statistics_service.yearly_rate_trend(
            self.state.final_data,
            region=self.selected_region,
        )

    def get_table_columns(self) -> list[str]:
        if self.state.final_data is None:
            return []
        return self._statistics_service.table_columns(self.state.final_data)

    def get_model_performance_rows(self) -> list[dict[str, float | str | None]]:
        return self._ai_service.get_model_performance_rows()

    def get_model_performance_summary(self) -> dict[str, float | str | None]:
        return self._ai_service.get_model_performance_summary()

    def _load_analysis_data(self, step: str, loader) -> pd.DataFrame | None:
        self.state = CrimeState(status=ProcessStatus.RUNNING, current_step=step)
        self._callback(self.state)
        self.analysis_messages = []

        try:
            df = loader()
        except Exception as exc:
            self._set_failed(step, str(exc))
            return None

        self.analysis_messages = list(self._analysis_data_service.last_messages)
        self.state.status = ProcessStatus.SUCCESS
        self.state.final_data = df
        self._callback(self.state)
        return df

    def _replace_state(self, state: CrimeState) -> None:
        self.state = state
        self._callback(self.state)

    def _set_failed(self, step: str, message: str) -> None:
        self.state.status = ProcessStatus.FAILED
        self.state.failed_step = step
        self.state.error_message = message
        self._callback(self.state)
