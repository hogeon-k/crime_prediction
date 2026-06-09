import pandas as pd
from pathlib import Path

import _path_setup  # pylint: disable=unused-import
from model.excel_model import ProcessResult, ProcessStatus
from viewmodel.crime_viewmodel import CrimeViewModel


class FakeCrimeService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def validate(self, df: pd.DataFrame) -> ProcessResult:
        self.calls.append("validate")
        return ProcessResult(True, "ok", df)

    def handle_missing(self, df: pd.DataFrame) -> ProcessResult:
        self.calls.append("handle_missing")
        return ProcessResult(True, "ok", df)

    def convert_types(self, df: pd.DataFrame) -> ProcessResult:
        self.calls.append("convert_types")
        return ProcessResult(True, "ok", df)


class FakeAIService:
    def __init__(self) -> None:
        self.target_year = None

    def predict_file(self, input_path: str, output_path: str, target_year: int):
        self.target_year = target_year
        return pd.DataFrame({"연도": [target_year]})

    @staticmethod
    def format_prediction_error(message: str) -> str:
        return message


def test_viewmodel_uses_injected_crime_service() -> None:
    service = FakeCrimeService()
    states = []
    vm = CrimeViewModel(callback=states.append, service=service)
    df = pd.DataFrame(
        {
            "범죄_유형": ["절도"],
            "지역": ["서울"],
            "연도": [2024],
            "발생_건수": [10],
            "인구수": [1_000_000],
        }
    )

    vm.process_from_df(df)

    assert service.calls == ["validate", "handle_missing", "convert_types"]
    assert vm.state.status == ProcessStatus.SUCCESS
    assert vm.state.final_data is not None
    assert states[-1].status == ProcessStatus.SUCCESS


def test_viewmodel_passes_target_year_to_ai_service() -> None:
    ai_service = FakeAIService()
    vm = CrimeViewModel(callback=lambda _state: None, ai_service=ai_service)

    result = vm.predict_file("input.csv", "output.csv", target_year=2025)

    assert ai_service.target_year == 2025
    assert result is not None
    assert result["연도"].iloc[0] == 2025
    assert vm.state.status == ProcessStatus.SUCCESS


def test_gui_files_do_not_import_ai_predict_directly() -> None:
    root = Path(__file__).resolve().parents[1]

    for relative_path in ("src/gui/excel_window.py", "run_app.py"):
        text = (root / relative_path).read_text(encoding="utf-8")
        assert "from ai.predict" not in text
        assert "import ai.predict" not in text
