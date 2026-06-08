import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
MODEL_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai.evaluator import ModelEvaluator
from ai.linear.linear_model import LinearRegressionModel
from ai.predict import predict
from ai.preprocessing import (
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TEST_YEAR,
    DEFAULT_TRAIN_YEARS,
    add_feature_engineering,
    build_feature_engineering_stats,
    split_train_test,
)
from ai.random_forest.random_forest import RandomForestRegressorModel
from ai.xgboost.xgb_model import XGBoostRegressorModel
from model.excel_model import UploadParams
from services.excel_pipeline import run_excel_pipeline


RANDOM_FOREST_CANDIDATES = {
    "random_forest_current": {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "random_forest_depth_8": {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "random_forest_depth_12": {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "random_forest_depth_12_leaf_2": {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
    },
    "random_forest_depth_16_leaf_2": {
        "n_estimators": 150,
        "max_depth": 16,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
    },
}


def create_models():
    models = {
        "linear": LinearRegressionModel(learning_rate=1e-16),
        "xgboost": XGBoostRegressorModel(),
    }

    for model_name, params in RANDOM_FOREST_CANDIDATES.items():
        models[model_name] = RandomForestRegressorModel(**params)

    return models


def train_models(X_train, y_train, feature_engineering_stats=None):
    models = create_models()
    feature_columns = list(X_train.columns)

    for model in models.values():
        model.fit(X_train, y_train)
        model.feature_columns = feature_columns
        model.feature_engineering_stats = feature_engineering_stats

    return models


def evaluate_models(models, X_test, y_test):
    results = {}

    for model_name, model in models.items():
        y_pred = predict(model, X_test)
        results[model_name] = {
            "model": model,
            "metrics": ModelEvaluator.evaluate(y_test, y_pred),
            "prediction_diversity": prediction_diversity_summary(y_pred),
        }

    return results


def prediction_diversity_summary(y_pred, top_n=5):
    series = pd.Series(y_pred, dtype=float)
    value_counts = series.value_counts(dropna=False)
    top_values = []

    for value, count in value_counts.head(top_n).items():
        top_values.append(
            {
                "value": float(value),
                "count": int(count),
                "ratio": float(count / len(series)) if len(series) else 0.0,
            }
        )

    return {
        "row_count": int(len(series)),
        "unique_count": int(series.nunique(dropna=False)),
        "unique_ratio": float(series.nunique(dropna=False) / len(series))
        if len(series)
        else 0.0,
        "min": float(series.min()) if len(series) else None,
        "max": float(series.max()) if len(series) else None,
        "mean": float(series.mean()) if len(series) else None,
        "std": float(series.std(ddof=0)) if len(series) else None,
        "top_values": top_values,
    }


def _sorted_years(values):
    return sorted(int(year) for year in values)


def train_and_evaluate(df):
    print("\n========== 전체 데이터 연도 분포 ==========")
    print(df["연도"].value_counts().sort_index())

    train_source_df = df[df["연도"].isin(DEFAULT_TRAIN_YEARS)]
    feature_engineering_stats = build_feature_engineering_stats(train_source_df)
    engineered_df = add_feature_engineering(df, stats=feature_engineering_stats)

    X_train, X_test, y_train, y_test = split_train_test(engineered_df)

    train_years = _sorted_years(X_train["연도"].unique())
    test_years = _sorted_years(X_test["연도"].unique())

    print("\n========== Train/Test 연도 확인 ==========")
    print(f"Train 연도: {train_years}")
    print(f"Test 연도 : {test_years}")

    assert set(train_years) == set(DEFAULT_TRAIN_YEARS), (
        "Train에 2022, 2023 외 연도가 포함됨"
    )
    assert set(test_years) == {DEFAULT_TEST_YEAR}, (
        "Test가 2024만으로 구성되지 않음"
    )

    models = train_models(
        X_train,
        y_train,
        feature_engineering_stats=feature_engineering_stats,
    )
    results = evaluate_models(models, X_test, y_test)
    best_name, best_result = ModelEvaluator.compare(results)

    return {
        "models": models,
        "results": results,
        "best_name": best_name,
        "best_model": best_result["model"],
        "train_years": train_years,
        "test_years": test_years,
        "feature_columns": list(X_train.columns),
        "feature_engineering": {
            "enabled": True,
            "columns": [
                "전년도_발생_건수",
                "전년도_범죄율",
                "지역별_평균_발생_건수",
                "범죄유형별_평균_발생_건수",
            ],
        },
    }


def print_train_report(results):
    ModelEvaluator.print_report(results)
    print_prediction_diversity_report(results)


def print_prediction_diversity_report(results):
    print("\n========== 예측 다양성 분석 ==========")

    for name, item in results.items():
        diversity = item.get("prediction_diversity", {})
        top_values = diversity.get("top_values", [])

        print(f"\n[{name}]")
        print(f"예측 행 수       : {diversity.get('row_count')}")
        print(f"고유 예측값 수   : {diversity.get('unique_count')}")
        print(f"고유 예측값 비율 : {diversity.get('unique_ratio', 0):.4f}")
        print(f"min / max        : {diversity.get('min')} / {diversity.get('max')}")
        print(f"mean / std       : {diversity.get('mean')} / {diversity.get('std')}")

        if top_values:
            print("가장 많이 반복되는 예측값:")
            for row in top_values:
                print(
                    f"  {row['value']:.6f} -> {row['count']}행 "
                    f"({row['ratio']:.2%})"
                )


def save_best_model(output, model_path=BEST_MODEL_PATH, info_path=MODEL_INFO_PATH):
    model_path = Path(model_path)
    info_path = Path(info_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    best_name = output["best_name"]
    best_model = output["best_model"]
    metrics = output["results"][best_name]["metrics"]

    print(f"\n========== 최종 저장 모델 ==========")
    print(f"Best Model: {best_name}")

    with model_path.open("wb") as file:
        pickle.dump(best_model, file)

    model_info = {
        "best_model": best_name,
        "metrics": {name: float(value) for name, value in metrics.items()},
        "prediction_diversity": output["results"][best_name][
            "prediction_diversity"
        ],
        "feature_columns": output["feature_columns"],
        "feature_engineering": output["feature_engineering"],
        "train_years": output["train_years"],
        "test_years": output["test_years"],
        "target_column": DEFAULT_TARGET_COLUMN,
    }

    with info_path.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, ensure_ascii=False, indent=2)

    print(f"Model saved: {model_path}")
    print(f"Model info saved: {info_path}")

    return model_path


def get_default_government_files():
    data_dir = SRC_DIR / "data"
    crime_files = sorted(data_dir.glob("crime_region_20*.csv"))
    pop_files = sorted(data_dir.glob("pop_20*.csv"))

    return [str(path) for path in crime_files], [str(path) for path in pop_files]


def make_training_dataframe(crime_files=None, pop_files=None):
    if crime_files is None or pop_files is None:
        default_crime_files, default_pop_files = get_default_government_files()
        crime_files = default_crime_files if crime_files is None else crime_files
        pop_files = default_pop_files if pop_files is None else pop_files

    params = UploadParams(
        mode="government",
        crime_files=list(crime_files),
        pop_files=list(pop_files),
    )

    return run_excel_pipeline(params)


def main():
    parser = argparse.ArgumentParser(
        description="정부 범죄/인구 CSV를 전처리한 뒤 발생 건수 예측 모델을 훈련합니다."
    )
    parser.add_argument(
        "--crime-files",
        nargs="+",
        help="범죄 CSV 파일 목록. 생략하면 src/data/crime_region_20*.csv를 사용합니다.",
    )
    parser.add_argument(
        "--pop-files",
        nargs="+",
        help="인구 CSV 파일 목록. 생략하면 src/data/pop_20*.csv를 사용합니다.",
    )
    args = parser.parse_args()

    df = make_training_dataframe(
        crime_files=args.crime_files,
        pop_files=args.pop_files,
    )

    print("\n========== 학습 데이터 정보 ==========")
    print(f"행 수: {df.shape[0]}")
    print(f"열 수: {df.shape[1]}")
    print(f"컬럼: {list(df.columns)}")

    output = train_and_evaluate(df)
    print_train_report(output["results"])
    save_best_model(output)


if __name__ == "__main__":
    main()
