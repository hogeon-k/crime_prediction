import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
MODEL_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai.evaluator import ModelEvaluator
from ai.experiments import (
    run_feature_importance_analysis,
    run_hyperparameter_experiments,
    run_year_walk_forward_validation,
    summarize_training_data,
)
from ai.linear.linear_model import LinearRegressionModel
from ai.predict import predict
from ai.preprocessing import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TRAINING_CONFIG,
    ENGINEERED_FEATURE_COLUMNS,
    add_feature_engineering,
    build_feature_engineering_stats,
    split_train_test,
)
from ai.random_forest.random_forest import RandomForestRegressorModel
from ai.xgboost.xgb_model import XGBoostRegressorModel
from model.excel_model import UploadParams
from services.excel_pipeline import run_excel_pipeline


class StandardizedLinearRegressionModel:
    """Scale model inputs before fitting the custom gradient-descent linear model."""

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.model = LinearRegressionModel(learning_rate=learning_rate, epochs=epochs)
        self.mean_ = None
        self.scale_ = None
        self.feature_columns = None
        self.feature_engineering_stats = None

    @staticmethod
    def _as_array(X):
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def fit(self, X, y):
        X_array = self._as_array(X)
        self.mean_ = X_array.mean(axis=0)
        self.scale_ = X_array.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.model.fit((X_array - self.mean_) / self.scale_, y)
        return self

    def predict(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("linear model scaler has not been fitted")
        X_array = self._as_array(X)
        return self.model.predict((X_array - self.mean_) / self.scale_)


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
        "linear": StandardizedLinearRegressionModel(learning_rate=0.01),
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
    return evaluate_models_train_test(models, X_test, y_test, X_test, y_test)


def evaluate_models_train_test(models, X_train, y_train, X_test, y_test, group_data=None):
    results = {}

    for model_name, model in models.items():
        y_train_pred = predict(model, X_train)
        y_pred = predict(model, X_test)
        results[model_name] = ModelEvaluator.evaluate_train_test(
            y_train,
            y_train_pred,
            y_test,
            y_pred,
            group_data=group_data,
        )
        results[model_name]["model"] = model

    return results


def prediction_diversity_summary(y_pred, top_n=5):
    return ModelEvaluator.prediction_diversity(y_pred, top_n=top_n)


def previous_year_baseline(df, config=DEFAULT_TRAINING_CONFIG):
    year_column, region_column, crime_column = DEFAULT_FEATURE_COLUMNS[:3]
    target_column = DEFAULT_TARGET_COLUMN

    train_df = df[df[year_column].isin(config.train_years)].copy()
    test_df = df[df[year_column] == config.test_year].copy()
    previous_year = config.test_year - 1

    previous_df = train_df[train_df[year_column] == previous_year]
    previous_lookup = (
        previous_df.groupby([region_column, crime_column])[target_column].mean().to_dict()
    )
    group_mean = train_df.groupby([region_column, crime_column])[target_column].mean().to_dict()
    global_mean = float(train_df[target_column].mean()) if not train_df.empty else 0.0

    predictions = []
    for _, row in test_df.iterrows():
        key = (row[region_column], row[crime_column])
        predictions.append(previous_lookup.get(key, group_mean.get(key, global_mean)))

    return {
        "name": "baseline_2023_same_region_crime",
        "metrics": ModelEvaluator.evaluate(test_df[target_column], predictions),
        "prediction_diversity": ModelEvaluator.prediction_diversity(predictions),
    }


def _selection_score(metrics):
    return (
        float(metrics["r2"]),
        -float(metrics["rmse"]),
        -float(metrics["mae"]),
    )


def _meets_or_beats_baseline(metrics, baseline_metrics):
    return (
        metrics["r2"] >= baseline_metrics["r2"]
        and metrics["rmse"] <= baseline_metrics["rmse"]
        and metrics["mae"] <= baseline_metrics["mae"]
    )


def _ablation_drop(full_metrics, ablation_metrics):
    return {
        "r2": float(full_metrics["r2"] - ablation_metrics["r2"]),
        "rmse": float(ablation_metrics["rmse"] - full_metrics["rmse"]),
        "mae": float(ablation_metrics["mae"] - full_metrics["mae"]),
    }


def select_best_model(
    results,
    baseline,
    ablation_results,
):
    baseline_metrics = baseline["metrics"]
    comparison = {
        "baseline": baseline,
        "ai_candidates": results,
        "baseline_warnings": {},
    }

    baseline_outperformed = True
    for name, item in results.items():
        metrics = item["test_metrics"]
        if not _meets_or_beats_baseline(metrics, baseline_metrics):
            comparison["baseline_warnings"][name] = {
                "metrics": metrics,
                "reason": "below baseline on at least one of R2, RMSE, MAE",
            }
        else:
            baseline_outperformed = False

    def sort_key(name):
        metrics = results[name]["test_metrics"]
        diversity = results[name].get("prediction_diversity", {})
        ablation_item = ablation_results.get(name)
        ablation_metrics = (
            ablation_item["test_metrics"] if ablation_item is not None else metrics
        )
        ablation_score = _selection_score(ablation_metrics)
        return (
            *_selection_score(metrics),
            float(diversity.get("unique_ratio", 0.0)),
            *ablation_score,
        )

    best_name = max(results, key=sort_key)
    best_result = results[best_name]
    ablation_item = ablation_results.get(best_name)
    best_baseline_name = baseline["name"]
    warnings = []

    if baseline_outperformed:
        warnings.extend(
            [
                "Baseline outperformed all AI models.",
                "However, project requires saving one AI model.",
                f"{best_name} was selected by Test R2, RMSE, and MAE among AI models.",
            ]
        )
    elif best_name in comparison["baseline_warnings"]:
        warnings.extend(
            [
                f"{best_name} is below baseline on at least one of R2, RMSE, MAE.",
                "However, project requires saving one AI model.",
            ]
        )

    reason = [
        "Selected by highest Test R2, then lower RMSE and MAE.",
        "Baseline is reported separately and is not saved as the final AI model.",
    ]

    reason.append(f"{best_name} had the best AI model selection score.")

    if ablation_item is not None:
        drop = _ablation_drop(best_result["test_metrics"], ablation_item["test_metrics"])
        reason.append(
            "Feature removal performance without previous-year features: "
            f"R2 drop={drop['r2']:.4f}, RMSE increase={drop['rmse']:.4f}, "
            f"MAE increase={drop['mae']:.4f}."
        )

    diversity = best_result.get("prediction_diversity", {})
    if diversity:
        reason.append(
            f"Better prediction diversity: unique ratio={diversity.get('unique_ratio', 0.0):.4f}."
        )

    return {
        "best_name": best_name,
        "best_baseline_name": best_baseline_name,
        "best_ai_name": best_name,
        "final_saved_model": best_name,
        "best_result": best_result,
        "best_model": best_result["model"],
        "best_is_baseline": False,
        "baseline_outperformed_ai": baseline_outperformed,
        "comparison": comparison,
        "warning": warnings,
        "reason": reason,
    }


def run_ablation_without_previous_features(
    engineered_df,
    config=DEFAULT_TRAINING_CONFIG,
    feature_engineering_stats=None,
):
    previous_features = set(ENGINEERED_FEATURE_COLUMNS[:2])
    ablation_features = DEFAULT_FEATURE_COLUMNS + [
        column
        for column in ENGINEERED_FEATURE_COLUMNS
        if column in engineered_df.columns and column not in previous_features
    ]

    X_train, X_test, y_train, y_test = split_train_test(
        engineered_df,
        feature_columns=ablation_features,
        train_years=config.train_years,
        test_year=config.test_year,
    )
    models = train_models(
        X_train,
        y_train,
        feature_engineering_stats=feature_engineering_stats,
    )

    test_group_data = engineered_df[engineered_df["연도"] == config.test_year]

    return evaluate_models_train_test(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        group_data=test_group_data,
    )


def _sorted_years(values):
    return sorted(int(year) for year in values)


def train_and_evaluate(df, config=DEFAULT_TRAINING_CONFIG):
    print("\n========== 전체 데이터 연도 분포 ==========")
    print(df["연도"].value_counts().sort_index())

    data_analysis = summarize_training_data(df)
    train_source_df = df[df["연도"].isin(config.train_years)]
    feature_engineering_stats = build_feature_engineering_stats(train_source_df)
    engineered_df = add_feature_engineering(df, stats=feature_engineering_stats)

    X_train, X_test, y_train, y_test = split_train_test(
        engineered_df,
        train_years=config.train_years,
        test_year=config.test_year,
    )

    train_years = _sorted_years(X_train["연도"].unique())
    test_years = _sorted_years(X_test["연도"].unique())

    print("\n========== Train/Test 연도 확인 ==========")
    print(f"Train 연도: {train_years}")
    print(f"Test 연도 : {test_years}")

    if set(train_years) != set(config.train_years):
        raise ValueError(
            f"Train 연도가 설정과 다릅니다. expected={list(config.train_years)}, actual={train_years}"
        )
    if set(test_years) != {config.test_year}:
        raise ValueError(
            f"Test 연도가 설정과 다릅니다. expected={[config.test_year]}, actual={test_years}"
        )

    models = train_models(
        X_train,
        y_train,
        feature_engineering_stats=feature_engineering_stats,
    )
    test_group_data = engineered_df[engineered_df["연도"] == config.test_year]
    results = evaluate_models_train_test(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        group_data=test_group_data,
    )
    baseline = previous_year_baseline(df, config=config)
    ablation_results = run_ablation_without_previous_features(
        engineered_df,
        config=config,
        feature_engineering_stats=feature_engineering_stats,
    )
    selection = select_best_model(results, baseline, ablation_results)
    hyperparameter_experiments = run_hyperparameter_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
    )
    year_validation = run_year_walk_forward_validation(df, config=config)
    feature_importance = run_feature_importance_analysis(
        selection["best_model"],
        X_test,
        y_test,
    )

    return {
        "models": models,
        "results": results,
        "baseline": baseline,
        "ablation_results": ablation_results,
        "hyperparameter_experiments": hyperparameter_experiments,
        "data_analysis": data_analysis,
        "year_validation": year_validation,
        "feature_importance": feature_importance,
        "selection": selection,
        "best_name": selection["best_name"],
        "best_model": selection["best_model"],
        "best_is_baseline": selection["best_is_baseline"],
        "best_baseline_name": selection["best_baseline_name"],
        "best_ai_name": selection["best_ai_name"],
        "final_saved_model": selection["final_saved_model"],
        "baseline_outperformed_ai": selection["baseline_outperformed_ai"],
        "selection_reason": selection["reason"],
        "selection_warning": selection["warning"],
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


def print_train_report(results, baseline=None, ablation_results=None, selection=None):
    ModelEvaluator.print_report(results)
    if baseline is not None:
        print_baseline_report(baseline)
    if ablation_results is not None:
        print_feature_removal_report(ablation_results)
    if selection is not None:
        print_selection_report(selection)


def print_extended_experiment_report(output):
    print_data_analysis_report(output.get("data_analysis", {}))
    print_hyperparameter_experiment_report(
        output.get("hyperparameter_experiments", {})
    )
    print_year_validation_report(output.get("year_validation", {}))
    print_feature_importance_report(output.get("feature_importance", {}))
    print_report_sentences()


def print_selection_report(selection):
    print("\n========== 최종 선택 모델 ==========")
    print(f"Best Baseline: {selection['best_baseline_name']}")
    print(f"Best AI Model: {selection['best_ai_name']}")
    print(f"Final Saved Model: {selection['final_saved_model']}")

    warnings = selection.get("warning", [])
    if warnings:
        print("\nWarning:")
        for warning in warnings:
            print(f"- {warning}")

    print("\nReason:")
    for reason in selection.get("reason", []):
        print(f"- {reason}")

    baseline_warnings = selection.get("comparison", {}).get("baseline_warnings", {})
    if baseline_warnings:
        print("\nAI models below baseline:")
        for name in baseline_warnings:
            print(f"- {name}: below baseline on R2, RMSE, or MAE")


def print_baseline_report(baseline):
    metrics = baseline["metrics"]
    diversity = baseline.get("prediction_diversity", {})

    print("\n========== Baseline 평가 ==========")
    print(f"\n[{baseline['name']}]")
    print(f"Test R2    : {metrics['r2']:.4f}")
    print(f"Test RMSE  : {metrics['rmse']:.4f}")
    print(f"Test MAE   : {metrics['mae']:.4f}")
    print(f"Test MSE   : {metrics['mse']:.4f}")
    if diversity:
        print(f"Unique prediction ratio: {diversity['unique_ratio']:.4f}")
        print("Top repeated predictions:")
        for row in diversity.get("top_values", []):
            print(
                f"  {row['value']:.6f} -> {row['count']}행 "
                f"({row['ratio']:.2%})"
            )


def print_feature_removal_report(results):
    print("\n========== Feature removal: without previous-year features ==========")

    for name, item in results.items():
        train_metrics = item["train_metrics"]
        test_metrics = item["test_metrics"]
        diversity = item.get("prediction_diversity", {})

        print(f"\n[feature_removal_without_previous_year_features] {name}")
        print(f"Train R2   : {train_metrics['r2']:.4f}")
        print(f"Train RMSE : {train_metrics['rmse']:.4f}")
        print(f"Train MAE  : {train_metrics['mae']:.4f}")
        print(f"Train MSE  : {train_metrics['mse']:.4f}")
        print(f"Test R2    : {test_metrics['r2']:.4f}")
        print(f"Test RMSE  : {test_metrics['rmse']:.4f}")
        print(f"Test MAE   : {test_metrics['mae']:.4f}")
        print(f"Test MSE   : {test_metrics['mse']:.4f}")
        if diversity:
            print(f"Unique prediction ratio: {diversity['unique_ratio']:.4f}")
            print("Top repeated predictions:")
            for row in diversity.get("top_values", []):
                print(
                    f"  {row['value']:.6f} -> {row['count']}행 "
                    f"({row['ratio']:.2%})"
                )


def print_ablation_report(results):
    print_feature_removal_report(results)


def _print_dataframe_table(rows, columns=None, float_format="{:.4f}".format):
    if not rows:
        print("(no rows)")
        return

    frame = pd.DataFrame(rows)
    if columns is not None:
        frame = frame[[column for column in columns if column in frame.columns]]
    print(frame.to_string(index=False, float_format=float_format))


def print_data_analysis_report(data_analysis):
    print("\n========== 데이터 분석 요약 ==========")

    distributions = data_analysis.get("distributions", {})
    rows = []
    for column, summary in distributions.items():
        rows.append({"column": column, **summary})
    print("\n[분포 및 IQR 이상값]")
    _print_dataframe_table(
        rows,
        columns=[
            "column",
            "count",
            "mean",
            "std",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "outlier_count",
        ],
    )

    print("\n[지역별 발생건수 상위 통계]")
    region_rows = [
        {"region": region, **stats}
        for region, stats in data_analysis.get("top_region_incidents", {}).items()
    ]
    _print_dataframe_table(region_rows)

    print("\n[범죄유형별 발생건수 상위 통계]")
    crime_rows = [
        {"crime_type": crime_type, **stats}
        for crime_type, stats in data_analysis.get("top_crime_type_incidents", {}).items()
    ]
    _print_dataframe_table(crime_rows)

    print("\n[숫자형 feature와 target 상관 Top 10]")
    _print_dataframe_table(
        data_analysis.get("correlations", [])[:10],
        columns=["feature", "correlation"],
    )


def print_hyperparameter_experiment_report(experiments):
    print("\n========== Hyperparameter experiment ==========")
    rows = []
    for row in experiments.get("rows", []):
        flat_row = {key: value for key, value in row.items() if key != "params"}
        flat_row.update(row.get("params", {}))
        rows.append(flat_row)

    _print_dataframe_table(
        rows,
        columns=[
            "family",
            "model",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "reg_lambda",
            "gamma",
            "train_r2",
            "test_r2",
            "r2_gap",
            "train_rmse",
            "test_rmse",
            "train_mae",
            "test_mae",
        ],
    )

    print("\nSelection rule:")
    print(f"- {experiments.get('selection_rule')}")

    best_by_family = experiments.get("best_by_family", {})
    if best_by_family:
        print("\nBest by family:")
        _print_dataframe_table(
            [
                {
                    "family": family,
                    "model": row["model"],
                    "test_r2": row["test_r2"],
                    "test_rmse": row["test_rmse"],
                    "test_mae": row["test_mae"],
                    "r2_gap": row["r2_gap"],
                }
                for family, row in best_by_family.items()
            ],
            columns=["family", "model", "test_r2", "test_rmse", "test_mae", "r2_gap"],
        )


def print_year_validation_report(year_validation):
    print("\n========== Year-based validation ==========")
    print(year_validation.get("strategy"))
    _print_dataframe_table(
        year_validation.get("folds", []),
        columns=[
            "fold",
            "model",
            "train_years",
            "validation_year",
            "train_r2",
            "validation_r2",
            "r2_gap",
            "validation_rmse",
            "validation_mae",
        ],
    )


def print_feature_importance_report(feature_importance):
    print("\n========== Feature importance ==========")

    print("\n[Split count importance Top 15]")
    _print_dataframe_table(
        feature_importance.get("split_importance", [])[:15],
        columns=["feature", "split_count", "importance"],
    )

    print("\n[Permutation importance Top 15]")
    _print_dataframe_table(
        feature_importance.get("permutation_importance", [])[:15],
        columns=["feature", "rmse_increase"],
    )


def print_report_sentences():
    print("\n========== 보고서 작성 문장 ==========")
    sentences = [
        "본 프로젝트는 연도 순서가 있는 2022~2024년 데이터를 사용하므로 무작위 K-Fold보다 과거 연도로 학습하고 다음 연도를 검증하는 연도 기반 Hold-out 및 Walk-Forward 검증이 더 적절하다.",
        "다만 관측 연도가 3개뿐이므로 Fold 1은 2022년 학습 후 2023년 검증, Fold 2는 2022~2023년 학습 후 2024년 검증으로 제한되며, 교차검증 결과는 최종 성능의 확정치가 아니라 일반화 가능성을 점검하는 보조 근거로 해석했다.",
        "RandomForest와 XGBoost는 각각 n_estimators, tree depth, split 조건, 정규화 관련 후보군을 두고 동일한 Train/Test 분리에서 R2, RMSE, MAE를 비교했다.",
        "최종 하이퍼파라미터 선택은 Test R2를 우선하고, 성능이 유사한 경우 Test RMSE와 Test MAE가 낮으며 Train/Test R2 차이가 작은 조합을 우선하는 기준으로 수행했다.",
        "전년도 발생건수 및 전년도 범죄율 feature를 제거한 실험은 엄밀한 의미의 모델 구성요소 ablation이라기보다 특정 feature군 제거 실험 또는 feature importance 검증으로 표현하는 것이 정확하다.",
        "발생건수, 인구수, 범죄율의 분포와 IQR 기준 극단값, 지역별 및 범죄유형별 발생건수 통계, 상관분석과 feature importance를 함께 확인하여 모델 성능뿐 아니라 데이터 구조와 주요 설명 변수를 점검했다.",
    ]
    for sentence in sentences:
        print(f"- {sentence}")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_best_model(output, model_path=BEST_MODEL_PATH, info_path=MODEL_INFO_PATH):
    model_path = Path(model_path)
    info_path = Path(info_path)

    best_name = output["final_saved_model"]
    best_model = output["models"][best_name]

    print(f"\n========== 최종 저장 모델 ==========")
    print(f"Final Saved Model: {best_name}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = output["results"][best_name]["metrics"]

    with model_path.open("wb") as file:
        pickle.dump(best_model, file)
    model_sha256 = _sha256_file(model_path)

    model_info = {
        "best_model": best_name,
        "best_baseline": output["best_baseline_name"],
        "best_ai_model": output["best_ai_name"],
        "final_saved_model": best_name,
        "model_file": model_path.name,
        "model_sha256": model_sha256,
        "metrics": {name: float(value) for name, value in metrics.items()},
        "prediction_diversity": output["results"][best_name][
            "prediction_diversity"
        ],
        "feature_columns": output["feature_columns"],
        "feature_engineering": output["feature_engineering"],
        "train_years": output["train_years"],
        "test_years": output["test_years"],
        "target_column": DEFAULT_TARGET_COLUMN,
        "selection_reason": output.get("selection_reason", []),
        "selection_warning": output.get("selection_warning", []),
        "baseline_outperformed_ai": bool(output.get("baseline_outperformed_ai")),
        "baseline_metrics": {
            name: float(value) for name, value in output["baseline"]["metrics"].items()
        },
    }

    with info_path.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, ensure_ascii=False, indent=2)

    print(f"Model saved: {model_path}")
    print(f"Model info saved: {info_path}")
    print(f"Model sha256: {model_sha256}")

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
    print_train_report(
        output["results"],
        baseline=output["baseline"],
        ablation_results=output["ablation_results"],
        selection=output["selection"],
    )
    print_extended_experiment_report(output)
    save_best_model(output)


if __name__ == "__main__":
    main()
