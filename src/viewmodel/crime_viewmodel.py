from __future__ import annotations

from typing import Callable

import pandas as pd

from model.excel_model import (
    CrimeState,
    ProcessStatus,
)
from services.crime_service import CrimeService


class CrimeViewModel:

    def __init__(self, callback: Callable[[CrimeState], None]) -> None:
        self._callback = callback
        self._service = CrimeService()
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

    def _set_failed(self, step: str, message: str) -> None:
        self.state.status = ProcessStatus.FAILED
        self.state.failed_step = step
        self.state.error_message = message
        self._callback(self.state)
