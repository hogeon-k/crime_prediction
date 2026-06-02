from __future__ import annotations
import threading
from typing import Callable
import pandas as pd

from model.excel_model import (
    ExcelResult,
    ExcelState,
    ProcessStatus,
    ColumnMapping,
    ValidationRule,
)
from services.excel_service import ExcelService


class ExcelViewModel:
    """
    MVVM - ViewModel
    ┌─────────────────────────────────────────────────────────┐
    │  쓰레드 구조                                             │
    │                                                         │
    │  Main thread          Daemon thread (Sub thread)        │
    │  ───────────          ─────────────────────────         │
    │  process() 호출  →   _run_pipeline() 실행               │
    │  즉시 반환            load → unify → missing → convert  │
    │                       단계마다 _notify() → View 갱신    │
    │                       실패 시 즉시 중단 (Fail-Fast)     │
    └─────────────────────────────────────────────────────────┘

    Daemon thread 선택 이유:
      - 메인 프로그램 종료 시 파이프라인이 자동으로 함께 종료
      - UI 가 닫혀도 백그라운드 작업이 프로세스를 붙들지 않음
    """

    _STEPS: list[tuple[str, str]] = [
        ("load_excel", "파일 로드"),
        ("unify_columns", "컬럼 통일"),
        ("handle_missing_values", "결측치 처리"),
        ("convert_types", "타입 변환"),
    ]

    def __init__(
        self,
        mapping: ColumnMapping | None = None,
        rule: ValidationRule | None = None,
        on_state_changed: Callable[[ExcelState], None] | None = None,
    ):
        self._service = ExcelService(mapping, rule)
        self._on_state_changed = on_state_changed or (lambda _: None)
        self._state = ExcelState()
        self._lock = threading.Lock()  # 상태 읽기/쓰기 보호

    # ── public ─────────────────────────────────────────────
    @property
    def state(self) -> ExcelState:
        with self._lock:
            return self._state

    def process(self, file_path: str) -> None:
        """
        Daemon Sub thread 로 파이프라인 실행.
        호출 즉시 반환 — UI 가 블로킹되지 않음.
        """
        self._reset(file_path)

        worker = threading.Thread(
            target=self._run_pipeline,
            args=(file_path,),
            daemon=True,  # 메인 종료 시 함께 종료
            name="ExcelPipeline",
        )
        worker.start()

    def process_sync(self, file_path: str) -> ExcelState:
        """
        동기 실행 (테스트 / CLI 용).
        쓰레드 없이 메인 쓰레드에서 직접 실행 후 최종 상태 반환.
        """
        self._reset(file_path)
        self._run_pipeline(file_path)
        return self.state

    # ── private ────────────────────────────────────────────
    def _run_pipeline(self, file_path: str) -> None:
        """Fail-Fast 파이프라인 — Sub thread 에서 실행"""
        df: pd.DataFrame | None = None

        for method_name, label in self._STEPS:
            self._update_step(label)

            result: ExcelResult = (
                getattr(self._service, method_name)(file_path)
                if method_name == "load_excel"
                else getattr(self._service, method_name)(df)
            )

            if not result.success:
                self._fail(label, result.message)
                return  # ← Fail-Fast: 이후 단계 실행 안 함

            df = result.data
            self._append_completed(label)

        self._succeed(df)

    def _reset(self, file_path: str) -> None:
        with self._lock:
            self._state = ExcelState(
                status=ProcessStatus.RUNNING,
                current_step=f"시작: {file_path}",
            )
        self._notify()

    def _update_step(self, label: str) -> None:
        with self._lock:
            self._state.current_step = label
        self._notify()

    def _append_completed(self, label: str) -> None:
        with self._lock:
            self._state.completed_steps.append(label)
        self._notify()

    def _fail(self, step: str, message: str) -> None:
        with self._lock:
            self._state.status = ProcessStatus.FAILED
            self._state.current_step = ""
            self._state.failed_step = step
            self._state.error_message = message
        self._notify()

    def _succeed(self, df: pd.DataFrame | None) -> None:
        with self._lock:
            self._state.status = ProcessStatus.SUCCESS
            self._state.current_step = ""
            self._state.final_data = df
        self._notify()

    def _notify(self) -> None:
        self._on_state_changed(self._state)
