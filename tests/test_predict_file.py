import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import pytest

from ai.predict import (
    DEFAULT_MODEL_PATH,
    PREDICTED_INCIDENTS_COLUMN,
    PREDICTED_RATE_COLUMN,
    prepare_prediction_features,
    predict_from_dataframe,
    predict_from_file,
)


def make_prediction_data():
    return pd.DataFrame(
        {
            "연도": [2024, 2024, 2025],
            "지역": ["서울", "부산", "인천"],
            "범죄_유형": ["절도", "폭력", "사기"],
            "인구수": [9300000, 3300000, 3000000],
        }
    )


def test_predict_from_dataframe_adds_prediction_columns():
    assert DEFAULT_MODEL_PATH.exists(), (
        "models/best_model.pkl이 없습니다. 먼저 python src/ai/train.py를 실행하세요."
    )

    result_df = predict_from_dataframe(make_prediction_data())

    assert PREDICTED_INCIDENTS_COLUMN in result_df.columns
    assert PREDICTED_RATE_COLUMN in result_df.columns
    assert len(result_df) == 3
    assert (result_df[PREDICTED_INCIDENTS_COLUMN] >= 0).all()
    assert (
        result_df[PREDICTED_RATE_COLUMN]
        == result_df[PREDICTED_INCIDENTS_COLUMN] / result_df["인구수"] * 100000
    ).all()


def test_predict_from_file_csv(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    make_prediction_data().to_csv(input_path, index=False, encoding="utf-8-sig")

    result_df = predict_from_file(input_path, output_path)
    saved_df = pd.read_csv(output_path)

    assert output_path.exists()
    assert PREDICTED_INCIDENTS_COLUMN in saved_df.columns
    assert PREDICTED_RATE_COLUMN in saved_df.columns
    assert len(result_df) == len(saved_df)


def test_predict_from_file_xlsx(tmp_path):
    input_path = tmp_path / "input.xlsx"
    output_path = tmp_path / "output.xlsx"
    make_prediction_data().to_excel(input_path, index=False, engine="openpyxl")

    result_df = predict_from_file(input_path, output_path)
    saved_df = pd.read_excel(output_path)

    assert output_path.exists()
    assert PREDICTED_INCIDENTS_COLUMN in saved_df.columns
    assert PREDICTED_RATE_COLUMN in saved_df.columns
    assert len(result_df) == len(saved_df)


def test_predict_from_dataframe_requires_feature_columns():
    df = make_prediction_data().drop(columns=["지역"])

    with pytest.raises(ValueError, match="예측에 필요한 컬럼"):
        predict_from_dataframe(df)


class DummyPredictionModel:
    feature_columns = [
        "연도",
        "인구수",
        "지역_서울",
        "지역_부산",
        "지역_인천",
        "범죄_유형_강도",
        "범죄_유형_절도범죄",
        "범죄_유형_폭력행위등",
    ]

    def predict(self, X):
        return (
            X["인구수"] / 100000
            + X["지역_서울"] * 10
            + X["지역_부산"] * 20
            + X["지역_인천"] * 30
            + X["범죄_유형_강도"] * 100
            + X["범죄_유형_절도범죄"] * 200
            + X["범죄_유형_폭력행위등"] * 300
        )


def make_alias_prediction_data():
    return pd.DataFrame(
        {
            "연도": [2025, 2025, 2025],
            "지역": ["서울특별시", "부산광역시", "인천"],
            "범죄_유형": ["절도", "폭력", "강도"],
            "인구수": [9000000, 3300000, 3000000],
        }
    )


def test_prepare_prediction_features_keeps_different_rows_after_encoding():
    encoded = prepare_prediction_features(
        DummyPredictionModel(),
        make_alias_prediction_data(),
    )

    assert list(encoded.columns) == DummyPredictionModel.feature_columns
    assert len(encoded.drop_duplicates()) == len(encoded)
    assert encoded["지역_서울"].tolist() == [1.0, 0.0, 0.0]
    assert encoded["지역_부산"].tolist() == [0.0, 1.0, 0.0]
    assert encoded["범죄_유형_절도범죄"].tolist() == [1.0, 0.0, 0.0]
    assert encoded["범죄_유형_폭력행위등"].tolist() == [0.0, 1.0, 0.0]


def test_predict_from_dataframe_does_not_force_same_prediction(monkeypatch):
    import ai.predict as predict_module

    monkeypatch.setattr(
        predict_module,
        "load_best_model",
        lambda model_path=predict_module.DEFAULT_MODEL_PATH: DummyPredictionModel(),
    )

    result_df = predict_from_dataframe(make_alias_prediction_data())

    assert result_df[PREDICTED_INCIDENTS_COLUMN].nunique() > 1


def test_predict_debug_prints_diagnostics(monkeypatch):
    import ai.predict as predict_module

    messages = []
    monkeypatch.setattr(
        predict_module,
        "load_best_model",
        lambda model_path=predict_module.DEFAULT_MODEL_PATH: DummyPredictionModel(),
    )

    predict_from_dataframe(
        make_alias_prediction_data(),
        debug=True,
        debug_printer=messages.append,
    )

    debug_text = "\n".join(str(message) for message in messages)
    assert "원본 입력 다양성" in debug_text
    assert "encoded feature 변화 요약" in debug_text
    assert "예측 결과 분포" in debug_text
    assert "입력은 다른데 예측이 거의 같은 행" in debug_text
    assert "Feature Importance" in debug_text
