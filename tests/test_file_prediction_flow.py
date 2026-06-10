import hashlib
import json
import pickle

import pandas as pd

from constants import (
    COL_CRIME_TYPE,
    COL_POPULATION,
    COL_REGION,
    COL_YEAR,
    PREDICTED_INCIDENTS_COLUMN,
    TARGET_YEAR_COLUMN,
)
from services.ai_service import AIService
from viewmodel.crime_viewmodel import CrimeViewModel


class ServicePredictionModel:
    feature_columns = [COL_YEAR, COL_POPULATION]

    @staticmethod
    def predict(X):
        return pd.Series(X[COL_POPULATION], dtype=float) / 100000


def make_prediction_input():
    return pd.DataFrame(
        {
            COL_YEAR: [2025, 2025],
            COL_REGION: ["서울", "부산"],
            COL_CRIME_TYPE: ["절도", "폭력"],
            COL_POPULATION: [9_000_000, 3_300_000],
        }
    )


def write_model(tmp_path):
    model_path = tmp_path / "best_model.pkl"
    info_path = tmp_path / "model_info.json"

    with model_path.open("wb") as file:
        pickle.dump(ServicePredictionModel(), file)

    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    info_path.write_text(
        json.dumps(
            {
                "best_model": "service_prediction_model",
                "model_file": model_path.name,
                "model_sha256": model_hash,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return model_path, info_path


def make_viewmodel(tmp_path):
    model_path, info_path = write_model(tmp_path)
    service = AIService(model_path=model_path, model_info_path=info_path)
    return CrimeViewModel(callback=lambda _state: None, ai_service=service)


def test_file_prediction_flow_success(tmp_path):
    viewmodel = make_viewmodel(tmp_path)
    input_path = tmp_path / "input.xlsx"
    output_path = tmp_path / "output.xlsx"
    make_prediction_input().to_excel(input_path, index=False, engine="openpyxl")

    result = viewmodel.predict_file(str(input_path), str(output_path), target_year=2025)

    assert result is not None
    assert output_path.exists()
    assert TARGET_YEAR_COLUMN in result.columns
    assert PREDICTED_INCIDENTS_COLUMN in result.columns
    assert result[PREDICTED_INCIDENTS_COLUMN].tolist() == [90.0, 33.0]


def test_file_prediction_flow_reports_missing_required_columns(tmp_path):
    viewmodel = make_viewmodel(tmp_path)
    input_path = tmp_path / "missing_region.csv"
    output_path = tmp_path / "output.csv"
    make_prediction_input().drop(columns=[COL_REGION]).to_csv(
        input_path,
        index=False,
        encoding="utf-8-sig",
    )

    result = viewmodel.predict_file(str(input_path), str(output_path), target_year=2025)

    assert result is None
    assert COL_REGION in viewmodel.state.error_message
    assert "필수 컬럼" in viewmodel.state.error_message


def test_file_prediction_flow_reports_missing_model(tmp_path):
    service = AIService(
        model_path=tmp_path / "missing_best_model.pkl",
        model_info_path=tmp_path / "missing_model_info.json",
    )
    viewmodel = CrimeViewModel(callback=lambda _state: None, ai_service=service)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    make_prediction_input().to_csv(input_path, index=False, encoding="utf-8-sig")

    result = viewmodel.predict_file(str(input_path), str(output_path), target_year=2025)

    assert result is None
    assert "모델 파일을 찾을 수 없습니다" in viewmodel.state.error_message
    assert "models/best_model.pkl" in viewmodel.state.error_message


def test_file_prediction_flow_reports_bad_extension(tmp_path):
    viewmodel = make_viewmodel(tmp_path)
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.xlsx"
    input_path.write_text("not a supported prediction file", encoding="utf-8")

    result = viewmodel.predict_file(str(input_path), str(output_path), target_year=2025)

    assert result is None
    assert "csv, xlsx, xls" in viewmodel.state.error_message


def test_model_performance_rows_are_loaded_through_viewmodel(tmp_path):
    viewmodel = make_viewmodel(tmp_path)

    rows = viewmodel.get_model_performance_rows()

    assert [row["model"] for row in rows] == ["Linear Regression"]
    assert all("rmse" in row for row in rows)
    assert all("inference_seconds" in row for row in rows)
    assert all("cpu_usage_percent" in row for row in rows)


def test_model_performance_summary_loads_model_info(tmp_path):
    info_path = tmp_path / "model_info.json"
    info_path.write_text(
        json.dumps(
            {
                "best_model": "linear",
                "metrics": {
                    "r2": 0.9513,
                    "rmse": 1799.14,
                    "mae": 748.26,
                    "mse": 3236904.5,
                    "smape": 12.4,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    service = AIService(
        model_path=tmp_path / "best_model.pkl",
        model_info_path=info_path,
    )

    summary = service.get_model_performance_summary()

    assert summary["model"] == "linear"
    assert summary["r2"] == 0.9513
    assert summary["rmse"] == 1799.14
    assert summary["message"] == ""


def test_model_performance_summary_handles_missing_file(tmp_path):
    service = AIService(
        model_path=tmp_path / "best_model.pkl",
        model_info_path=tmp_path / "missing_model_info.json",
    )

    summary = service.get_model_performance_summary()

    assert summary["model"] is None
    assert "모델 성능 정보 파일을 찾을 수 없습니다" in summary["message"]
