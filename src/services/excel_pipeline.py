from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from model.excel_model import ProcessStatus, UploadParams
from services.crime_service import CrimeService
from viewmodel.crime_viewmodel import CrimeViewModel


def run_excel_pipeline(
    params: UploadParams,
    on_state_update: Callable[[Any], None] | None = None,
) -> pd.DataFrame:
    errors = params.validation_errors()
    if errors:
        raise ValueError("\n".join(errors))

    callback = on_state_update if on_state_update is not None else (lambda _s: None)
    vm = CrimeViewModel(callback=callback)
    service = CrimeService()

    if params.mode == "standard":
        result = service.load_uploaded(params.standard_file)
        if not result.success:
            raise ValueError(result.message)
        vm.process_from_df(result.data)
    else:
        vm.process(
            crime_files=params.crime_files,
            pop_files=params.pop_files,
        )

    if vm.state.status != ProcessStatus.SUCCESS:
        raise RuntimeError(
            f"pipeline failed [{vm.state.failed_step}]: {vm.state.error_message}"
        )

    if vm.state.final_data is None:
        raise RuntimeError("no result data")

    return vm.state.final_data
