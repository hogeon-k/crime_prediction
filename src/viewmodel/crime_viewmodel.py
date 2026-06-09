from __future__ import annotations

from typing import Callable

import pandas as pd

from model.excel_model import (
    CrimeState,
    ProcessStatus,
    UploadParams,
)
from services.ai_service import AIService
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
        statistics_service: StatisticsService | None = None,
    ) -> None:
        self._callback = callback
        self._service = service or CrimeService()
        self._ai_service = ai_service or AIService()
        self._statistics_service = statistics_service or StatisticsService()
        self.state = CrimeState()

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

    def predict_one(
        self,
        year: int,
        region: str,
        crime_type: str,
        population: int,
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

    def _replace_state(self, state: CrimeState) -> None:
        self.state = state
        self._callback(self.state)

    def _set_failed(self, step: str, message: str) -> None:
        self.state.status = ProcessStatus.FAILED
        self.state.failed_step = step
        self.state.error_message = message
        self._callback(self.state)
