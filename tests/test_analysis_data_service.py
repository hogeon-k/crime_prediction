import pandas as pd

import _path_setup  # pylint: disable=unused-import
from constants import (
    COL_CRIME_RATE,
    COL_CRIME_TYPE,
    COL_INCIDENTS,
    COL_POPULATION,
    COL_REGION,
    COL_YEAR,
    PREDICTED_INCIDENTS_COLUMN,
    PREDICTED_RATE_COLUMN,
)
from services.analysis_data_service import (
    ACTUAL_KIND,
    DATA_KIND_COLUMN,
    PREDICTED_KIND,
    AnalysisDataService,
)


def make_actual_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_YEAR: [2022, 2023, 2024],
            COL_REGION: ["서울", "서울", "서울"],
            COL_CRIME_TYPE: ["절도", "절도", "절도"],
            COL_INCIDENTS: [100, 120, 140],
            COL_POPULATION: [9_000_000, 9_000_000, 9_000_000],
            COL_CRIME_RATE: [1.1, 1.3, 1.5],
        }
    )


def make_prediction_df(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_YEAR: [year],
            COL_REGION: ["서울"],
            COL_CRIME_TYPE: ["절도"],
            COL_POPULATION: [9_000_000],
            PREDICTED_INCIDENTS_COLUMN: [160.0 + (year - 2025) * 10],
            PREDICTED_RATE_COLUMN: [1.7 + (year - 2025) * 0.1],
        }
    )


def make_service(tmp_path, prediction_years=(2025, 2026, 2027)) -> AnalysisDataService:
    actual_path = tmp_path / "src" / "data" / "processed_crime_data.csv"
    actual_path.parent.mkdir(parents=True)
    make_actual_df().to_csv(actual_path, index=False, encoding="utf-8-sig")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for year in prediction_years:
        make_prediction_df(year).to_excel(
            data_dir / f"prediction_result_{year}.xlsx",
            index=False,
            engine="openpyxl",
        )

    return AnalysisDataService(root_dir=tmp_path, actual_data_path=actual_path)


def test_prediction_result_path_uses_year(tmp_path) -> None:
    service = AnalysisDataService(root_dir=tmp_path)

    assert service.prediction_result_path(2025) == tmp_path / "data" / "prediction_result_2025.xlsx"
    assert service.prediction_result_path(2026) == tmp_path / "data" / "prediction_result_2026.xlsx"
    assert service.prediction_result_path(2027) == tmp_path / "data" / "prediction_result_2027.xlsx"


def test_combined_actual_prediction_data_has_2022_to_2027(tmp_path) -> None:
    service = make_service(tmp_path)

    combined = service.load_combined_actual_prediction_data()

    assert sorted(combined[COL_YEAR].unique().tolist()) == [2022, 2023, 2024, 2025, 2026, 2027]
    assert combined[DATA_KIND_COLUMN].tolist()[:3] == [ACTUAL_KIND, ACTUAL_KIND, ACTUAL_KIND]
    assert set(combined[DATA_KIND_COLUMN].tail(3)) == {PREDICTED_KIND}


def test_combined_data_allows_missing_prediction_file(tmp_path) -> None:
    service = make_service(tmp_path, prediction_years=(2025, 2026))

    combined = service.load_combined_actual_prediction_data()

    assert sorted(combined[COL_YEAR].unique().tolist()) == [2022, 2023, 2024, 2025, 2026]
    assert "2027 예측 결과 파일이 없습니다." in service.last_messages


def test_yearly_crime_rate_summary_groups_by_year_and_kind(tmp_path) -> None:
    service = make_service(tmp_path)
    combined = service.load_combined_actual_prediction_data()

    summary = service.get_yearly_crime_rate_summary(combined)

    assert summary[[COL_YEAR, DATA_KIND_COLUMN]].values.tolist() == [
        [2022, ACTUAL_KIND],
        [2023, ACTUAL_KIND],
        [2024, ACTUAL_KIND],
        [2025, PREDICTED_KIND],
        [2026, PREDICTED_KIND],
        [2027, PREDICTED_KIND],
    ]
    assert summary[COL_CRIME_RATE].notna().all()
