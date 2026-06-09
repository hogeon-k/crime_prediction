import pandas as pd

from ai.train import RANDOM_FOREST_CANDIDATES, print_train_report, train_and_evaluate


def make_sample_data():
    return pd.DataFrame(
        {
            "연도": [2022, 2022, 2023, 2023, 2024, 2024],
            "지역": ["서울", "부산", "서울", "인천", "부산", "서울"],
            "범죄_유형": ["절도", "폭력", "절도", "사기", "폭력", "절도"],
            "인구수": [9500000, 3400000, 9400000, 3000000, 3300000, 9300000],
            "발생_건수": [100, 80, 120, 95, 90, 130],
            "범죄율": [1.05, 2.35, 1.27, 3.17, 2.73, 1.39],
        }
    )


def test_train_and_evaluate():
    # Small sample data is only for smoke testing the training flow.
    # Do not use this result to judge real model performance.
    df = make_sample_data()
    output = train_and_evaluate(df)

    expected_models = {"linear", "xgboost", *RANDOM_FOREST_CANDIDATES}

    assert set(output["models"]) == expected_models
    assert set(output["results"]) == expected_models
    assert output["baseline"]["name"] == "baseline_2023_same_region_crime"
    assert output["best_name"] in output["models"]
    assert output["best_ai_name"] in output["models"]
    assert output["final_saved_model"] in output["models"]
    assert output["final_saved_model"] == "random_forest_depth_8"
    assert output["best_model"] is output["models"][output["final_saved_model"]]
    assert output["best_is_baseline"] is False
    assert output["selection_reason"]

    for item in output["results"].values():
        assert set(item["metrics"]) == {"mse", "rmse", "mae", "smape", "r2"}
        assert set(item["prediction_diversity"]) == {
            "row_count",
            "unique_count",
            "unique_ratio",
            "min",
            "max",
            "mean",
            "std",
            "top_values",
        }


def main():
    output = train_and_evaluate(make_sample_data())
    print_train_report(output["results"])


if __name__ == "__main__":
    main()
