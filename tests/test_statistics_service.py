import pandas as pd

import _path_setup  # pylint: disable=unused-import
from constants import PREDICTED_INCIDENTS_COLUMN, PREDICTED_RATE_COLUMN
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
