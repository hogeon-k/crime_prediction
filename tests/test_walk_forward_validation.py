import pandas as pd

import ai.experiments as experiments
from ai.experiments import run_year_walk_forward_validation


def make_walk_forward_data():
    rows = []
    for year in range(2018, 2025):
        for region in ("서울", "부산"):
            for crime_type in ("절도", "폭력"):
                base = 100 if region == "서울" else 70
                crime_offset = 20 if crime_type == "절도" else 5
                incidents = base + crime_offset + (year - 2018) * 3
                population = 9_000_000 if region == "서울" else 3_300_000
                rows.append(
                    {
                        "연도": year,
                        "지역": region,
                        "범죄_유형": crime_type,
                        "인구수": population,
                        "발생_건수": incidents,
                        "범죄율": incidents / population * 100_000,
                    }
                )
    return pd.DataFrame(rows)


def test_walk_forward_folds_are_generated_from_available_years():
    result = run_year_walk_forward_validation(make_walk_forward_data())
    folds = result["folds"]
    validation_years = sorted({row["validation_year"] for row in folds})

    assert result["available_years"] == [2019, 2020, 2021, 2022, 2023, 2024]
    assert validation_years == [2021, 2022, 2023, 2024]
    assert {row["model"] for row in folds} == {"linear", "random_forest", "xgboost"}
    assert result["averages"].keys() == {"linear", "random_forest", "xgboost"}


def test_walk_forward_validation_year_is_after_train_years():
    result = run_year_walk_forward_validation(make_walk_forward_data())

    for row in result["folds"]:
        assert max(row["train_years"]) < row["validation_year"]
        assert row["train_row_count"] > 0
        assert row["validation_row_count"] > 0
        assert {"validation_r2", "validation_rmse", "validation_mae", "validation_mse"} <= row.keys()


def test_walk_forward_feature_stats_do_not_use_validation_target(monkeypatch):
    recorded_max_years = []
    original = experiments.build_feature_engineering_stats

    def spy(train_source_df, *args, **kwargs):
        recorded_max_years.append(int(train_source_df["연도"].max()))
        return original(train_source_df, *args, **kwargs)

    monkeypatch.setattr(experiments, "build_feature_engineering_stats", spy)

    result = run_year_walk_forward_validation(make_walk_forward_data())
    validation_years = [row["validation_year"] for row in result["folds"][::3]]

    assert recorded_max_years == [year - 1 for year in validation_years]
