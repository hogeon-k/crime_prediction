from __future__ import annotations

from typing import Callable

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

        # ✅ 개선: 1단계는 파일 경로를 받으므로 별도 처리, 이후 단계는 df를 전달
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

            # ✅ 추가: 1단계 이외에서 df가 None이면 파이프라인 설계 오류
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

    def _set_failed(self, step: str, message: str) -> None:
        """✅ 추가: 실패 상태 설정 및 콜백 호출을 한 곳에서 처리"""
        self.state.status = ProcessStatus.FAILED
        self.state.failed_step = step
        self.state.error_message = message
        self._callback(self.state)
