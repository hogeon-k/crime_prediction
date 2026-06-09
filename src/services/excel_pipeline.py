from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from model.excel_model import CrimeState, ProcessStatus, UploadParams
from services.crime_service import CrimeService


def _emit_state(
    callback: Callable[[Any], None],
    state: CrimeState,
    current_step: str,
) -> None:
    state.current_step = current_step
    callback(state)


def run_excel_pipeline(
    params: UploadParams,
    on_state_update: Callable[[Any], None] | None = None,
) -> pd.DataFrame:
    errors = params.validation_errors()
    if errors:
        raise ValueError("\n".join(errors))

    callback = on_state_update if on_state_update is not None else (lambda _s: None)
    service = CrimeService()
    state = CrimeState(status=ProcessStatus.RUNNING)

    if params.mode == "standard":
        _emit_state(callback, state, "업로드 로드")
        result = service.load_uploaded(params.standard_file)
        if not result.success:
            raise ValueError(result.message)
        df = result.data
    else:
        _emit_state(callback, state, "데이터 병합")
        result = service.load_and_merge(
            crime_files=params.crime_files,
            pop_files=params.pop_files,
        )
        if not result.success:
            raise ValueError(result.message)
        df = result.data

    steps = [
        ("검증", service.validate),
        ("결측치 처리", service.handle_missing),
        ("타입 변환", service.convert_types),
    ]

    for name, fn in steps:
        if df is None:
            raise RuntimeError("no result data")
        state.completed_steps.append(state.current_step)
        _emit_state(callback, state, name)
        result = fn(df)
        if not result.success:
            state.status = ProcessStatus.FAILED
            state.failed_step = name
            state.error_message = result.message
            callback(state)
            raise ValueError(result.message)
        df = result.data

    if df is None:
        raise RuntimeError("no result data")

    state.completed_steps.append(state.current_step)
    state.status = ProcessStatus.SUCCESS
    state.final_data = df
    callback(state)

    return df
