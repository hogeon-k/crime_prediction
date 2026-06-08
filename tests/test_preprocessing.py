import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import pytest

from ai.preprocessing import split_features_target, split_by_year, split_train_test


def make_sample_data():
    return pd.DataFrame(
        {
            "연도": [2022, 2023, 2024],
            "지역": ["서울", "부산", "서울"],
            "범죄_유형": ["절도", "폭력", "절도"],
            "인구수": [9500000, 9400000, 9300000],
            "발생_건수": [100, 120, 130],
            "범죄율": [1.05, 1.27, 1.39],
        }
    )


def test_split_features_target():
    df = make_sample_data()

    X, y = split_features_target(df)

    assert "발생_건수" not in X.columns
    assert "지역" not in X.columns
    assert "범죄_유형" not in X.columns
    assert "연도" in X.columns
    assert "인구수" in X.columns
    assert "지역_서울" in X.columns
    assert "범죄_유형_절도" in X.columns
    assert list(y) == [100, 120, 130]


def test_split_by_year():
    df = make_sample_data()

    X_train, X_test, y_train, y_test = split_by_year(
        df,
        test_year=2024,
    )

    assert list(X_train["연도"]) == [2022, 2023]
    assert list(X_test["연도"]) == [2024]
    assert list(y_train) == [100, 120]
    assert list(y_test) == [130]


def test_split_train_test_uses_year_holdout():
    df = make_sample_data()

    X_train, X_test, y_train, y_test = split_train_test(df)

    assert sorted(X_train["연도"].unique()) == [2022, 2023]
    assert sorted(X_test["연도"].unique()) == [2024]
    assert 2024 not in set(X_train["연도"])
    assert not set(X_test["연도"]).intersection({2022, 2023})
    assert list(y_train) == [100, 120]
    assert list(y_test) == [130]
    assert list(X_train.columns) == list(X_test.columns)


def test_split_by_year_no_test_data():
    df = make_sample_data()

    with pytest.raises(ValueError):
        split_by_year(df, test_year=2025)
