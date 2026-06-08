import pandas as pd

DEFAULT_FEATURE_COLUMNS = ["연도", "지역", "범죄_유형", "인구수"]
DEFAULT_TARGET_COLUMN = "발생_건수"
DEFAULT_TRAIN_YEARS = (2022, 2023)
DEFAULT_TEST_YEAR = 2024

_REGION_ALIASES = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "강원도": "강원",
    "강원특별자치도": "강원",
    "충청북도": "충북",
    "충청남도": "충남",
    "전라북도": "전북",
    "전북특별자치도": "전북",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
}

_CRIME_ALIASES = {
    "절도": "절도범죄",
    "마약": "마약범죄",
    "도박": "도박범죄",
    "교통": "교통범죄",
    "선거": "선거범죄",
    "보건": "보건범죄",
    "환경": "환경범죄",
    "노동": "노동범죄",
    "병역": "병역범죄",
    "안보": "안보범죄",
    "성풍속": "성풍속범죄",
    "기타": "기타범죄",
    "폭력": "폭력행위등",
}


def _known_category_values(
    encoded_columns: list[str] | None,
    prefix: str,
) -> set[str]:
    if encoded_columns is None:
        return set()

    marker = f"{prefix}_"
    return {
        column.removeprefix(marker)
        for column in encoded_columns
        if column.startswith(marker)
    }


def _normalize_text_value(value) -> str:
    return " ".join(str(value).strip().split())


def _normalize_region_value(value, known_regions: set[str]) -> str:
    normalized = _normalize_text_value(value)
    normalized = _REGION_ALIASES.get(normalized, normalized)

    if normalized in known_regions:
        return normalized

    compact = normalized.replace(" ", "")
    for region in known_regions:
        if compact.startswith(region):
            return region

    return normalized


def _normalize_crime_value(value, known_crime_types: set[str]) -> str:
    normalized = _normalize_text_value(value)
    normalized = _CRIME_ALIASES.get(normalized, normalized)

    if normalized in known_crime_types:
        return normalized

    candidates = [
        f"{normalized}범죄",
        normalized.replace("/", " "),
        normalized.replace("/", ""),
        normalized.replace(" ", ""),
    ]
    for candidate in candidates:
        if candidate in known_crime_types:
            return candidate

    return normalized


def normalize_prediction_features(
    X: pd.DataFrame,
    encoded_columns: list[str] | None = None,
) -> pd.DataFrame:
    normalized = X.copy()
    known_regions = _known_category_values(encoded_columns, "지역")
    known_crime_types = _known_category_values(encoded_columns, "범죄_유형")

    if "지역" in normalized.columns:
        normalized["지역"] = normalized["지역"].map(
            lambda value: _normalize_region_value(value, known_regions)
        )

    if "범죄_유형" in normalized.columns:
        normalized["범죄_유형"] = normalized["범죄_유형"].map(
            lambda value: _normalize_crime_value(value, known_crime_types)
        )

    return normalized


def encode_features(
    X: pd.DataFrame,
    encoded_columns: list[str] | None = None,
) -> pd.DataFrame:
    categorical_columns = [
        column for column in ("지역", "범죄_유형") if column in X.columns
    ]
    encoded_X = pd.get_dummies(X, columns=categorical_columns, dtype=float)

    if encoded_columns is not None:
        encoded_X = encoded_X.reindex(columns=encoded_columns, fill_value=0.0)

    return encoded_X


def split_features_target(
    df: pd.DataFrame,
    feature_columns=None,
    target_column: str = DEFAULT_TARGET_COLUMN,
):
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS

    missing_columns = [
        col for col in feature_columns + [target_column] if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    X = encode_features(df[feature_columns])
    y = df[target_column]

    return X, y


def split_train_test(
    df: pd.DataFrame,
    feature_columns=None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    train_years: tuple[int, ...] = DEFAULT_TRAIN_YEARS,
    test_year: int = DEFAULT_TEST_YEAR,
):
    train_df = df[df["연도"].isin(train_years)]
    test_df = df[df["연도"] == test_year]

    if train_df.empty:
        raise ValueError(f"{list(train_years)} 학습 데이터가 없습니다.")

    if test_df.empty:
        raise ValueError(f"{test_year} 테스트 데이터가 없습니다.")

    X, y = split_features_target(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    X_train = X.loc[train_df.index]
    y_train = y.loc[train_df.index]

    X_test = X.loc[test_df.index]
    y_test = y.loc[test_df.index]

    return X_train, X_test, y_train, y_test


def split_by_year(
    df: pd.DataFrame,
    test_year: int,
    feature_columns=None,
    target_column: str = DEFAULT_TARGET_COLUMN,
):
    """
    연도 기준 분리

    예:
        train = 2022~2023
        test  = 2024
    """

    train_df = df[df["연도"] < test_year]
    test_df = df[df["연도"] == test_year]

    if train_df.empty:
        raise ValueError(f"{test_year} 이전 학습 데이터가 없습니다.")

    if test_df.empty:
        raise ValueError(f"{test_year} 테스트 데이터가 없습니다.")

    X, y = split_features_target(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    X_train = X.loc[train_df.index]
    y_train = y.loc[train_df.index]

    X_test = X.loc[test_df.index]
    y_test = y.loc[test_df.index]

    return X_train, X_test, y_train, y_test
