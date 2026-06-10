from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai.evaluator import ModelEvaluator
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
from constants import (
    COL_CRIME_RATE,
    COL_CRIME_TYPE,
    COL_INCIDENTS,
    COL_POPULATION,
    COL_REGION,
    COL_YEAR,
)


RF_EXPERIMENT_GRID = [
    {
        "n_estimators": 30,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
    },
    {
        "n_estimators": 80,
        "max_depth": 8,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 120,
        "max_depth": 12,
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "max_features": "log2",
    },
    {
        "n_estimators": 150,
        "max_depth": 16,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": None,
    },
]

XGB_EXPERIMENT_GRID = [
    {
        "n_estimators": 30,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 2,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
    {
        "n_estimators": 50,
        "learning_rate": 0.08,
        "max_depth": 3,
        "min_samples_split": 4,
        "reg_lambda": 2.0,
        "gamma": 0.0,
    },
    {
        "n_estimators": 80,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_samples_split": 4,
        "reg_lambda": 3.0,
        "gamma": 0.1,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_samples_split": 6,
        "reg_lambda": 5.0,
        "gamma": 0.2,
    },
    {
        "n_estimators": 120,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_samples_split": 8,
        "reg_lambda": 8.0,
        "gamma": 0.5,
    },
]


@dataclass(frozen=True)
class ExperimentCandidate:
    name: str
    family: str
    params: dict
    factory: Callable[[], object]


def _resolve_max_features(value, n_features: int):
    if value == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if value == "log2":
        return max(1, int(np.log2(n_features)))
    return value


def _rf_candidates(n_features: int) -> list[ExperimentCandidate]:
    candidates = []
    for index, params in enumerate(RF_EXPERIMENT_GRID, start=1):
        resolved_params = {
            **params,
            "max_features": _resolve_max_features(params["max_features"], n_features),
        }
        display_params = dict(params)
        candidates.append(
            ExperimentCandidate(
                name=f"rf_grid_{index}",
                family="RandomForest",
                params=display_params,
                factory=lambda resolved_params=resolved_params: RandomForestRegressorModel(
                    **resolved_params
                ),
            )
        )
    return candidates


def _xgb_candidates() -> list[ExperimentCandidate]:
    candidates = []
    for index, params in enumerate(XGB_EXPERIMENT_GRID, start=1):
        candidates.append(
            ExperimentCandidate(
                name=f"xgb_grid_{index}",
                family="XGBoost",
                params=dict(params),
                factory=lambda params=params: XGBoostRegressorModel(**params),
            )
        )
    return candidates


def _metric_row(candidate: ExperimentCandidate, model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_metrics = ModelEvaluator.evaluate(y_train, train_pred)
    test_metrics = ModelEvaluator.evaluate(y_test, test_pred)

    return {
        "model": candidate.name,
        "family": candidate.family,
        "params": candidate.params,
        "train_r2": float(train_metrics["r2"]),
        "test_r2": float(test_metrics["r2"]),
        "r2_gap": float(train_metrics["r2"] - test_metrics["r2"]),
        "train_rmse": float(train_metrics["rmse"]),
        "test_rmse": float(test_metrics["rmse"]),
        "train_mae": float(train_metrics["mae"]),
        "test_mae": float(test_metrics["mae"]),
    }


def run_hyperparameter_experiments(X_train, y_train, X_test, y_test) -> dict:
    candidates = [*_rf_candidates(X_train.shape[1]), *_xgb_candidates()]
    rows = []

    for candidate in candidates:
        model = candidate.factory()
        model.fit(X_train, y_train)
        rows.append(_metric_row(candidate, model, X_train, y_train, X_test, y_test))

    best_by_family = {}
    for family in ("RandomForest", "XGBoost"):
        family_rows = [row for row in rows if row["family"] == family]
        best_by_family[family] = min(
            family_rows,
            key=lambda row: (
                -row["test_r2"],
                row["test_rmse"],
                row["test_mae"],
                max(row["r2_gap"], 0.0),
            ),
        )

    best_overall = min(
        rows,
        key=lambda row: (
            -row["test_r2"],
            row["test_rmse"],
            row["test_mae"],
            max(row["r2_gap"], 0.0),
        ),
    )

    return {
        "rows": rows,
        "best_by_family": best_by_family,
        "best_overall": best_overall,
        "selection_rule": (
            "Test R2를 우선 최대화하고, 동률 또는 근소한 차이에서는 Test RMSE, "
            "Test MAE, Train/Test R2 gap 순서로 더 작은 조합을 선택한다."
        ),
    }


def _numeric_summary(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "q1": q1,
        "median": float(numeric.median()),
        "q3": q3,
        "max": float(numeric.max()),
        "iqr": float(iqr),
        "outlier_count": int(((numeric < lower) | (numeric > upper)).sum()),
        "outlier_lower": float(lower),
        "outlier_upper": float(upper),
    }


def _correlation_summary(df: pd.DataFrame, target_column: str) -> list[dict]:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_column not in numeric_df.columns:
        return []
    correlations = numeric_df.corr(numeric_only=True)[target_column].drop(labels=[target_column])
    correlations = correlations.dropna().sort_values(key=lambda values: values.abs(), ascending=False)
    return [
        {"feature": feature, "correlation": float(value)}
        for feature, value in correlations.items()
    ]


def summarize_training_data(df: pd.DataFrame) -> dict:
    distribution_columns = [COL_INCIDENTS, COL_POPULATION, COL_CRIME_RATE]
    distributions = {
        column: _numeric_summary(df[column])
        for column in distribution_columns
        if column in df.columns
    }

    region_stats = {}
    if {COL_REGION, COL_INCIDENTS}.issubset(df.columns):
        region_stats = (
            df.groupby(COL_REGION)[COL_INCIDENTS]
            .agg(["count", "mean", "median", "min", "max", "sum"])
            .sort_values("sum", ascending=False)
            .head(10)
            .to_dict("index")
        )

    crime_type_stats = {}
    if {COL_CRIME_TYPE, COL_INCIDENTS}.issubset(df.columns):
        crime_type_stats = (
            df.groupby(COL_CRIME_TYPE)[COL_INCIDENTS]
            .agg(["count", "mean", "median", "min", "max", "sum"])
            .sort_values("sum", ascending=False)
            .head(10)
            .to_dict("index")
        )

    return {
        "distributions": distributions,
        "top_region_incidents": region_stats,
        "top_crime_type_incidents": crime_type_stats,
        "correlations": _correlation_summary(df, COL_INCIDENTS),
    }


def _split_feature_importance(model, feature_names: list[str]) -> list[dict]:
    if not hasattr(model, "trees"):
        return []

    counts_by_feature = dict.fromkeys(feature_names, 0)

    def visit(node):
        if node is None or getattr(node, "value", None) is not None:
            return
        feature_index = getattr(node, "feature_index", None)
        if feature_index is not None and 0 <= feature_index < len(feature_names):
            counts_by_feature[feature_names[feature_index]] += 1
        visit(getattr(node, "left", None))
        visit(getattr(node, "right", None))

    for tree in getattr(model, "trees", []):
        visit(getattr(tree, "root", None))

    total = sum(counts_by_feature.values())
    if total == 0:
        return []

    rows = [
        {
            "feature": feature,
            "split_count": int(split_count),
            "importance": float(split_count / total),
        }
        for feature, split_count in counts_by_feature.items()
        if split_count > 0
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def permutation_importance(
    model,
    X: pd.DataFrame,
    y,
    n_repeats: int = 3,
    random_state: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(random_state)
    baseline_pred = model.predict(X)
    baseline_rmse = ModelEvaluator.rmse(y, baseline_pred)
    rows = []

    for column in X.columns:
        increases = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            values = shuffled[column].to_numpy(copy=True)
            rng.shuffle(values)
            shuffled[column] = values
            shuffled_pred = model.predict(shuffled)
            increases.append(ModelEvaluator.rmse(y, shuffled_pred) - baseline_rmse)
        rows.append(
            {
                "feature": column,
                "rmse_increase": float(np.mean(increases)),
            }
        )

    return sorted(rows, key=lambda row: row["rmse_increase"], reverse=True)


def run_feature_importance_analysis(best_model, X_test: pd.DataFrame, y_test) -> dict:
    return {
        "split_importance": _split_feature_importance(best_model, list(X_test.columns)),
        "permutation_importance": permutation_importance(best_model, X_test, y_test),
    }


def run_year_walk_forward_validation(
    df: pd.DataFrame,
    config=DEFAULT_TRAINING_CONFIG,
) -> dict:
    years = sorted(int(year) for year in pd.Series(df[COL_YEAR]).dropna().unique())
    folds = []

    for fold_index, validation_year in enumerate(years[1:], start=1):
        train_years = tuple(year for year in years if year < validation_year)
        if not train_years:
            continue

        train_source_df = df[df[COL_YEAR].isin(train_years)]
        stats = build_feature_engineering_stats(train_source_df)
        engineered_df = add_feature_engineering(df, stats=stats)
        X_train, X_valid, y_train, y_valid = split_train_test(
            engineered_df,
            train_years=train_years,
            test_year=validation_year,
        )

        candidates = {
            "random_forest": RandomForestRegressorModel(
                n_estimators=100,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features=_resolve_max_features("sqrt", X_train.shape[1]),
            ),
            "xgboost": XGBoostRegressorModel(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=4,
                reg_lambda=3.0,
                gamma=0.1,
            ),
        }

        for model_name, model in candidates.items():
            model.fit(X_train, y_train)
            valid_pred = model.predict(X_valid)
            train_pred = model.predict(X_train)
            train_metrics = ModelEvaluator.evaluate(y_train, train_pred)
            valid_metrics = ModelEvaluator.evaluate(y_valid, valid_pred)
            folds.append(
                {
                    "fold": fold_index,
                    "model": model_name,
                    "train_years": list(train_years),
                    "validation_year": validation_year,
                    "train_r2": float(train_metrics["r2"]),
                    "validation_r2": float(valid_metrics["r2"]),
                    "r2_gap": float(train_metrics["r2"] - valid_metrics["r2"]),
                    "validation_rmse": float(valid_metrics["rmse"]),
                    "validation_mae": float(valid_metrics["mae"]),
                }
            )

    return {
        "strategy": (
            "연도 순서가 있는 데이터이므로 일반 K-Fold보다 과거 연도로 학습하고 "
            "다음 연도를 검증하는 Walk-Forward 검증을 사용한다. 현재 데이터가 "
            "2022~2024 3개 연도뿐이라 fold 수는 제한적이며, 결과는 보조 근거로 해석한다."
        ),
        "folds": folds,
    }
