import pandas as pd
import pytest

import _path_setup  # pylint: disable=unused-import
from ai.predict import (
    POPULATION_METHOD_CARRY_FORWARD,
    POPULATION_METHOD_UPLOADED,
    PREDICTION_METHOD_COLUMN,
    PREDICTION_STEP_COLUMN,
    PREVIOUS_INCIDENTS_COLUMN,
    PREVIOUS_RATE_COLUMN,
    build_initial_recursive_input,
    build_next_year_input,
    predict_recursive,
    predict_recursive_from_file,
)
from constants import (
    BASE_YEAR_COLUMN,
    COL_CRIME_RATE,
    COL_CRIME_TYPE,
    COL_INCIDENTS,
    COL_POPULATION,
    COL_REGION,
    COL_YEAR,
    PREDICTED_INCIDENTS_COLUMN,
    PREDICTED_RATE_COLUMN,
    TARGET_YEAR_COLUMN,
)


class RecursiveMockModel:
    feature_columns = [
        COL_YEAR,
        COL_POPULATION,
        PREVIOUS_INCIDENTS_COLUMN,
        PREVIOUS_RATE_COLUMN,
    ]

    @staticmethod
    def predict(X):
        return X[PREVIOUS_INCIDENTS_COLUMN] + 10.0


def make_base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_YEAR: [2024, 2024],
            COL_REGION: ["서울", "부산"],
            COL_CRIME_TYPE: ["절도", "폭력"],
            COL_POPULATION: [1_000_000, 2_000_000],
            COL_INCIDENTS: [100.0, 200.0],
            COL_CRIME_RATE: [10.0, 10.0],
        }
    )


def by_key(df: pd.DataFrame, region: str, crime_type: str) -> pd.Series:
    rows = df[(df[COL_REGION] == region) & (df[COL_CRIME_TYPE] == crime_type)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_recursive_prediction_links_previous_features_across_years() -> None:
    results = predict_recursive(make_base_df(), 2025, 2027, model=RecursiveMockModel())

    row_2025 = by_key(results[2025], "서울", "절도")
    row_2026 = by_key(results[2026], "서울", "절도")
    row_2027 = by_key(results[2027], "서울", "절도")

    assert row_2025[PREVIOUS_INCIDENTS_COLUMN] == 100.0
    assert row_2025[PREVIOUS_RATE_COLUMN] == 10.0
    assert row_2026[PREVIOUS_INCIDENTS_COLUMN] == row_2025[PREDICTED_INCIDENTS_COLUMN]
    assert row_2026[PREVIOUS_RATE_COLUMN] == row_2025[PREDICTED_RATE_COLUMN]
    assert row_2027[PREVIOUS_INCIDENTS_COLUMN] == row_2026[PREDICTED_INCIDENTS_COLUMN]
    assert row_2027[PREVIOUS_RATE_COLUMN] == row_2026[PREDICTED_RATE_COLUMN]
    assert row_2026[BASE_YEAR_COLUMN] == 2025
    assert row_2027[BASE_YEAR_COLUMN] == 2026
    assert row_2027[PREDICTION_STEP_COLUMN] == 3
    assert row_2027[PREDICTION_METHOD_COLUMN] == "recursive"


def test_recursive_prediction_preserves_one_to_one_keys() -> None:
    results = predict_recursive(make_base_df(), 2025, 2027, model=RecursiveMockModel())

    base_keys = set(map(tuple, make_base_df()[[COL_REGION, COL_CRIME_TYPE]].to_numpy()))
    for df in results.values():
        keys = set(map(tuple, df[[COL_REGION, COL_CRIME_TYPE]].to_numpy()))
        assert keys == base_keys
        assert len(df) == len(base_keys)


def test_initial_recursive_input_requires_expected_base_year() -> None:
    bad = make_base_df().assign(**{COL_YEAR: [2023, 2024]})

    with pytest.raises(ValueError, match="최신 실제 연도"):
        build_initial_recursive_input(bad, 2025)


def test_recursive_prediction_rejects_duplicate_keys() -> None:
    duplicated = pd.concat([make_base_df(), make_base_df().head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="중복"):
        predict_recursive(duplicated, 2025, 2025, model=RecursiveMockModel())


def test_explicit_previous_features_are_not_overwritten() -> None:
    input_df = build_initial_recursive_input(make_base_df(), 2025)
    input_df.loc[input_df[COL_REGION] == "서울", PREVIOUS_INCIDENTS_COLUMN] = 1234.0
    prediction = build_next_year_input(
        pd.DataFrame(
            {
                COL_YEAR: [2025],
                COL_REGION: ["서울"],
                COL_CRIME_TYPE: ["절도"],
                COL_POPULATION: [1_000_000],
                PREDICTED_INCIDENTS_COLUMN: [333.0],
                PREDICTED_RATE_COLUMN: [33.3],
                PREDICTION_STEP_COLUMN: [1],
            }
        ),
        2026,
        population_source_df=make_base_df(),
    )

    assert input_df.loc[input_df[COL_REGION] == "서울", PREVIOUS_INCIDENTS_COLUMN].iloc[0] == 1234.0
    assert prediction[PREVIOUS_INCIDENTS_COLUMN].iloc[0] == 333.0
    assert prediction[PREVIOUS_RATE_COLUMN].iloc[0] == 33.3


def test_recursive_population_policy_uses_future_population_when_available() -> None:
    base = pd.concat(
        [
            make_base_df(),
            pd.DataFrame(
                {
                    COL_YEAR: [2025],
                    COL_REGION: ["서울"],
                    COL_CRIME_TYPE: ["절도"],
                    COL_POPULATION: [1_100_000],
                    COL_INCIDENTS: [None],
                    COL_CRIME_RATE: [None],
                }
            ),
        ],
        ignore_index=True,
    )

    initial = build_initial_recursive_input(base, 2025)
    row = by_key(initial, "서울", "절도")

    assert row[COL_POPULATION] == 1_100_000
    assert row["인구수_추정_방법"] == POPULATION_METHOD_UPLOADED


def test_recursive_population_policy_carry_forward_when_future_missing() -> None:
    initial = build_initial_recursive_input(make_base_df(), 2025)

    assert set(initial["인구수_추정_방법"]) == {POPULATION_METHOD_CARRY_FORWARD}


def test_recursive_prediction_rejects_missing_population() -> None:
    bad = make_base_df()
    bad.loc[0, COL_POPULATION] = None

    with pytest.raises(ValueError, match="인구수가 누락"):
        predict_recursive(bad, 2025, 2025, model=RecursiveMockModel())


def test_recursive_prediction_saves_yearly_files(tmp_path) -> None:
    input_path = tmp_path / "base.xlsx"
    output_dir = tmp_path / "data"
    make_base_df().to_excel(input_path, index=False, engine="openpyxl")

    results = predict_recursive_from_file(
        input_path,
        output_dir,
        start_year=2025,
        end_year=2027,
        model=RecursiveMockModel(),
    )

    assert sorted(results) == [2025, 2026, 2027]
    for year in results:
        path = output_dir / f"prediction_result_{year}.xlsx"
        assert path.exists()
        saved = pd.read_excel(path)
        assert (saved[TARGET_YEAR_COLUMN] == year).all()
        assert PREVIOUS_INCIDENTS_COLUMN in saved.columns
        assert PREVIOUS_RATE_COLUMN in saved.columns

    predict_recursive_from_file(
        input_path,
        output_dir,
        start_year=2025,
        end_year=2025,
        model=RecursiveMockModel(),
    )
    assert (output_dir / "prediction_result_2025_backup_1.xlsx").exists()
