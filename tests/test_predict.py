import pandas as pd
import pytest

from ai.predict import DEFAULT_MODEL_PATH, predict, predict_one
from ai.preprocessing import add_feature_engineering


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


class BasicPredictionModel:
    feature_columns = ["연도", "인구수"]

    @staticmethod
    def predict(X):
        return X["인구수"] / 100000


class SinglePredictionRangeModel:
    feature_columns = [
        "연도",
        "인구수",
        "전년도_발생_건수",
        "전년도_범죄율",
        "지역별_평균_발생_건수",
        "범죄유형별_평균_발생_건수",
        "지역_서울",
        "범죄_유형_절도범죄",
    ]
    feature_engineering_stats = {
        "global_mean_incidents": 100.0,
        "region_mean_incidents": {"서울": 1000.0},
        "crime_type_mean_incidents": {"절도범죄": 2000.0},
        "region_crime_mean_incidents": {("서울", "절도범죄"): 5000.0},
        "region_mean_population": {"서울": 10_000_000.0},
        "region_population_bounds": {"서울": {"min": 9_000_000.0, "max": 10_000_000.0}},
    }

    @staticmethod
    def predict(X):
        return X["전년도_발생_건수"]


@pytest.mark.integration
def test_best_model_file_exists():
    assert DEFAULT_MODEL_PATH.exists(), (
        "models/best_model.pkl이 없습니다. 먼저 python src/ai/train.py를 실행하세요."
    )


@pytest.mark.integration
def test_predict_with_saved_model():
    df = make_sample_data()

    y_pred = predict(df)

    assert len(y_pred) == len(df)


def test_predict():
    df = make_sample_data()

    y_pred = predict(BasicPredictionModel(), df[["연도", "인구수"]])

    assert len(y_pred) == len(df)


def test_predict_one():
    result = predict_one(
        year=2025,
        region="서울",
        crime_type="절도",
        population=9000000,
        model=SinglePredictionRangeModel(),
    )

    assert isinstance(result, float)


def test_predict_one_rejects_unrealistic_region_population():
    with pytest.raises(ValueError, match="지역별 인구 범위"):
        predict_one(
            year=2026,
            region="서울",
            crime_type="절도",
            population=1_000_000,
            model=SinglePredictionRangeModel(),
        )


def test_prediction_fallback_incidents_scale_with_population():
    df = pd.DataFrame(
        {
            "연도": [2026, 2026],
            "지역": ["서울", "서울"],
            "범죄_유형": ["절도범죄", "절도범죄"],
            "인구수": [10_000_000, 5_000_000],
        }
    )

    engineered = add_feature_engineering(
        df,
        stats=SinglePredictionRangeModel.feature_engineering_stats,
    )

    assert engineered["전년도_발생_건수"].tolist() == [5000.0, 2500.0]


def test_negative_prediction_is_clipped_to_zero():
    df = pd.DataFrame({"연도": [2025], "인구수": [9000000]})

    y_pred = predict(NegativePredictionModel(), df)

    assert y_pred == [0.0]


def test_predict_without_model():
    with pytest.raises(ValueError):
        predict(None, [[2025, 9000000]])
