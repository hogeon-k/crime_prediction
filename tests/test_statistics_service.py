import pandas as pd

import _path_setup  # pylint: disable=unused-import
from constants import PREDICTED_INCIDENTS_COLUMN, PREDICTED_RATE_COLUMN
from services.analysis_data_service import ACTUAL_KIND, DATA_KIND_COLUMN, PREDICTED_KIND
from services.statistics_service import StatisticsService


def make_prediction_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "연도": [2025, 2025, 2025],
            "지역": ["서울", "서울", "부산"],
            "범죄_유형": ["절도", "폭력", "절도"],
            "인구수": [9_000_000, 9_000_000, 3_300_000],
            PREDICTED_INCIDENTS_COLUMN: [100.0, 300.0, 50.0],
            PREDICTED_RATE_COLUMN: [1.1, 3.3, 1.5],
        }
    )


def test_prediction_summary() -> None:
    summary = StatisticsService.prediction_summary(make_prediction_result())

    assert summary["row_count"] == 3
    assert summary["column_count"] == 6
    assert summary["avg_predicted_incidents"] == 150.0
    assert summary["max_predicted_incidents"] == 300.0
    assert summary["avg_predicted_rate"] == 1.9666666666666668
    assert summary["region_count"] == 2
    assert summary["crime_type_count"] == 2
    assert summary["year_min"] == 2025
    assert summary["year_max"] == 2025


def test_group_by_region() -> None:
    rows = StatisticsService.group_by_region(make_prediction_result())

    assert rows == [
        {"label": "서울", "value": 400.0},
        {"label": "부산", "value": 50.0},
    ]


def test_group_by_crime_type() -> None:
    rows = StatisticsService.group_by_crime_type(make_prediction_result())

    assert rows == [
        {"label": "폭력", "value": 300.0},
        {"label": "절도", "value": 150.0},
    ]


def test_top_region_by_crime_rate() -> None:
    rows = StatisticsService.top_region_by_crime_rate(make_prediction_result(), n=1)

    assert rows == [{"label": "서울", "value": 2.2}]


def test_filter_records_by_region_year_and_sort() -> None:
    filtered = StatisticsService.filter_records(
        make_prediction_result(),
        region="서울",
        year=2025,
        crime_type_query="절도",
        sort_by=PREDICTED_INCIDENTS_COLUMN,
    )

    assert len(filtered) == 1
    assert filtered.iloc[0][PREDICTED_INCIDENTS_COLUMN] == 100.0


def test_yearly_rate_trend() -> None:
    df = pd.concat(
        [
            make_prediction_result(),
            make_prediction_result().assign(
                연도=2026,
                **{PREDICTED_RATE_COLUMN: [2.0, 4.0, 1.0]},
            ),
        ],
        ignore_index=True,
    )

    rows = StatisticsService.yearly_rate_trend(df, region="서울")

    assert rows == [
        {"year": 2025, "crime_rate": 2.2},
        {"year": 2026, "crime_rate": 3.0},
    ]


def test_yearly_rate_summary_filters_specific_region() -> None:
    df = pd.concat(
        [
            make_prediction_result().assign(**{DATA_KIND_COLUMN: PREDICTED_KIND}),
            pd.DataFrame(
                {
                    "연도": [2024],
                    "지역": ["부산"],
                    "범죄_유형": ["절도"],
                    "인구수": [3_300_000],
                    "발생_건수": [40],
                    "범죄율": [1.2],
                    DATA_KIND_COLUMN: [ACTUAL_KIND],
                }
            ),
        ],
        ignore_index=True,
    )

    summary = StatisticsService.yearly_rate_summary(df, region="부산")

    assert summary[["연도", DATA_KIND_COLUMN, "범죄율"]].values.tolist() == [
        [2024, ACTUAL_KIND, 1.2],
        [2025, PREDICTED_KIND, 1.5],
    ]


def test_yearly_rate_summary_uses_all_regions_when_region_is_none() -> None:
    summary = StatisticsService.yearly_rate_summary(make_prediction_result())

    assert summary["연도"].tolist() == [2025]
    assert summary.iloc[0]["범죄율"] == make_prediction_result()[PREDICTED_RATE_COLUMN].mean()


def test_yearly_rate_summary_returns_empty_for_no_match() -> None:
    summary = StatisticsService.yearly_rate_summary(make_prediction_result(), region="제주")

    assert summary.empty


def test_yearly_rate_summary_sorts_dynamic_years() -> None:
    df = pd.concat(
        [
            make_prediction_result().assign(연도=2027),
            make_prediction_result().assign(연도=2025),
            make_prediction_result().assign(연도=2026),
        ],
        ignore_index=True,
    )

    summary = StatisticsService.yearly_rate_summary(df)

    assert summary["연도"].tolist() == [2025, 2026, 2027]
