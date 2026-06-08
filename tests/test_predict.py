# pylint: disable=wrong-import-position

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import pytest

from ai.predict import DEFAULT_MODEL_PATH, predict, predict_one


def make_sample_data():
    return pd.DataFrame(
        {
            "연도": [2024, 2024, 2024],
            "지역": ["서울", "부산", "인천"],
            "범죄_유형": ["절도", "폭력", "사기"],
            "인구수": [9300000, 3300000, 3000000],
        }
    )


class NegativePredictionModel:
    feature_columns = ["연도", "인구수"]

    @staticmethod
    def predict(_X):
        return [-10.0]


def test_best_model_file_exists():
    assert DEFAULT_MODEL_PATH.exists(), (
        "models/best_model.pkl이 없습니다. 먼저 python src/ai/train.py를 실행하세요."
    )


def test_predict():
    df = make_sample_data()

    y_pred = predict(df)

    assert len(y_pred) == len(df)


def test_predict_one():
    result = predict_one(
        year=2025,
        region="서울",
        crime_type="절도",
        population=9000000,
    )

    assert isinstance(result, float)


def test_negative_prediction_is_clipped_to_zero():
    df = pd.DataFrame({"연도": [2025], "인구수": [9000000]})

    y_pred = predict(NegativePredictionModel(), df)

    assert y_pred == [0.0]


def test_predict_without_model():
    with pytest.raises(ValueError):
        predict(None, [[2025, 9000000]])
