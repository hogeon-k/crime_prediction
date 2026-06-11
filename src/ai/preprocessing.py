from dataclasses import dataclass

import pandas as pd

from constants import (
    COL_CRIME_RATE,
    COL_CRIME_TYPE,
    COL_INCIDENTS,
    COL_POPULATION,
    COL_REGION,
    COL_YEAR,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMN,
)


@dataclass(frozen=True)
class TrainingConfig:
    train_years: tuple[int, ...] | None = None
    test_year: int | None = None
    exclude_first_year_for_previous_features: bool = True
    valid_years: tuple[int, ...] = tuple(range(2018, 2027))


DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_TRAIN_YEARS = DEFAULT_TRAINING_CONFIG.train_years
DEFAULT_TEST_YEAR = DEFAULT_TRAINING_CONFIG.test_year
CRIME_RATE_COLUMN = COL_CRIME_RATE
ENGINEERED_FEATURE_COLUMNS = [
    "전년도_발생_건수",
    "전년도_범죄율",
    "지역별_평균_발생_건수",
    "범죄유형별_평균_발생_건수",
]


def sorted_years(df: pd.DataFrame) -> list[int]:
    years = pd.to_numeric(df[COL_YEAR], errors="coerce").dropna().astype(int)
    return sorted(years.unique().tolist())


def feature_available_years(
    df: pd.DataFrame,
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> list[int]:
    years = sorted_years(df)
    if config.exclude_first_year_for_previous_features and len(years) > 1:
        return years[1:]
    return years


def excluded_previous_feature_years(
    df: pd.DataFrame,
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> list[int]:
    years = sorted_years(df)
    available = set(feature_available_years(df, config=config))
    return [year for year in years if year not in available]


def resolve_train_test_years(
    df: pd.DataFrame,
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> tuple[tuple[int, ...], int]:
    available_years = feature_available_years(df, config=config)
    if len(available_years) < 2:
        raise ValueError("학습/테스트 분리를 위해 최소 2개 이상의 사용 가능 연도가 필요합니다.")

    test_year = config.test_year if config.test_year is not None else max(available_years)
    if test_year not in available_years:
        raise ValueError(f"테스트 연도 {test_year}가 사용 가능 연도에 없습니다: {available_years}")

    if config.train_years is None:
        train_years = tuple(year for year in available_years if year < test_year)
    else:
        train_years = tuple(year for year in config.train_years if year in available_years)

    if not train_years:
        raise ValueError(f"테스트 연도 {test_year} 이전 학습 가능 연도가 없습니다.")
    if any(year >= test_year for year in train_years):
        raise ValueError(f"Train 연도는 Test 연도보다 과거여야 합니다: train={train_years}, test={test_year}")

    return train_years, int(test_year)

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
        normalized.replace("·", ""),
        normalized.replace("ㆍ", ""),
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


def unknown_prediction_categories(
    X: pd.DataFrame,
    encoded_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    if encoded_columns is None:
        return {}

    unknown: dict[str, list[str]] = {}
    checks = {
        "지역": _known_category_values(encoded_columns, "지역"),
        "범죄_유형": _known_category_values(encoded_columns, "범죄_유형"),
    }

    for column, known_values in checks.items():
        if column not in X.columns or not known_values:
            continue
        values = set(X[column].dropna().astype(str))
        missing = sorted(values - known_values)
        if missing:
            unknown[column] = missing

    return unknown


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


def fit_feature_encoder(X: pd.DataFrame) -> list[str]:
    """학습 데이터에만 one-hot encoder 컬럼 목록을 맞춘다."""
    return list(encode_features(X).columns)


def transform_features(X: pd.DataFrame, encoded_columns: list[str]) -> pd.DataFrame:
    """학습 데이터에서 fit한 컬럼 구조로 입력 데이터를 변환한다."""
    return encode_features(X, encoded_columns=encoded_columns)


def build_feature_engineering_stats(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> dict:
    population_column = DEFAULT_FEATURE_COLUMNS[3]
    required_columns = {COL_REGION, COL_CRIME_TYPE, population_column, target_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"feature engineering 필수 컬럼이 없습니다: {sorted(missing_columns)}")

    target = pd.to_numeric(df[target_column], errors="coerce")
    population = pd.to_numeric(df[population_column], errors="coerce")
    global_mean = float(target.mean()) if not target.dropna().empty else 0.0

    return {
        "global_mean_incidents": global_mean,
        "region_mean_incidents": (
            df.assign(_target=target).groupby(COL_REGION)["_target"].mean().to_dict()
        ),
        "crime_type_mean_incidents": (
            df.assign(_target=target).groupby(COL_CRIME_TYPE)["_target"].mean().to_dict()
        ),
        "region_crime_mean_incidents": (
            df.assign(_target=target)
            .groupby([COL_REGION, COL_CRIME_TYPE])["_target"]
            .mean()
            .to_dict()
        ),
        "region_mean_population": (
            df.assign(_population=population).groupby(COL_REGION)["_population"].mean().to_dict()
        ),
        "region_population_bounds": (
            df.assign(_population=population)
            .groupby(COL_REGION)["_population"]
            .agg(["min", "max"])
            .to_dict("index")
        ),
    }


def add_feature_engineering(
    df: pd.DataFrame,
    stats: dict | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    engineered = df.copy()

    if stats is None and target_column in engineered.columns:
        stats = build_feature_engineering_stats(engineered, target_column=target_column)
    if stats is None:
        stats = {
            "global_mean_incidents": 0.0,
            "region_mean_incidents": {},
            "crime_type_mean_incidents": {},
            "region_crime_mean_incidents": {},
            "region_mean_population": {},
            "region_population_bounds": {},
        }

    global_mean = float(stats.get("global_mean_incidents", 0.0))
    region_means = stats.get("region_mean_incidents", {})
    crime_type_means = stats.get("crime_type_mean_incidents", {})
    region_crime_means = stats.get("region_crime_mean_incidents", {})
    region_population_means = stats.get("region_mean_population", {})
    population_column = DEFAULT_FEATURE_COLUMNS[3]

    engineered["지역별_평균_발생_건수"] = engineered[COL_REGION].map(region_means)
    engineered["범죄유형별_평균_발생_건수"] = engineered[COL_CRIME_TYPE].map(crime_type_means)

    fallback_key_values = []
    for _, row in engineered.iterrows():
        region = row[COL_REGION]
        crime_type = row[COL_CRIME_TYPE]
        fallback_value = region_crime_means.get((region, crime_type), global_mean)
        mean_population = region_population_means.get(region)

        if population_column in engineered.columns and mean_population:
            population_value = pd.to_numeric(row[population_column], errors="coerce")
            if pd.notna(population_value) and mean_population > 0:
                fallback_value *= float(population_value) / float(mean_population)

        fallback_key_values.append(fallback_value)
    engineered["전년도_발생_건수"] = fallback_key_values
    engineered["전년도_범죄율"] = 0.0

    if {target_column, CRIME_RATE_COLUMN}.issubset(engineered.columns):
        sort_columns = [COL_REGION, COL_CRIME_TYPE, COL_YEAR]
        sorted_df = engineered.sort_values(sort_columns)
        grouped = sorted_df.groupby([COL_REGION, COL_CRIME_TYPE], sort=False)
        prev_incidents = grouped[target_column].shift(1)
        prev_rate = grouped[CRIME_RATE_COLUMN].shift(1)
        engineered.loc[sorted_df.index, "전년도_발생_건수"] = prev_incidents.values
        engineered.loc[sorted_df.index, "전년도_범죄율"] = prev_rate.values

    engineered["전년도_발생_건수"] = engineered["전년도_발생_건수"].fillna(
        pd.Series(fallback_key_values, index=engineered.index)
    )
    engineered["전년도_범죄율"] = engineered["전년도_범죄율"].fillna(0.0)
    engineered["지역별_평균_발생_건수"] = engineered["지역별_평균_발생_건수"].fillna(global_mean)
    engineered["범죄유형별_평균_발생_건수"] = engineered[
        "범죄유형별_평균_발생_건수"
    ].fillna(global_mean)

    for column in ENGINEERED_FEATURE_COLUMNS:
        engineered[column] = pd.to_numeric(engineered[column], errors="coerce").fillna(0.0)

    return engineered


def split_features_target(
    df: pd.DataFrame,
    feature_columns=None,
    target_column: str = DEFAULT_TARGET_COLUMN,
):
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS + [
            column for column in ENGINEERED_FEATURE_COLUMNS if column in df.columns
        ]

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
    train_years: tuple[int, ...] | None = DEFAULT_TRAIN_YEARS,
    test_year: int | None = DEFAULT_TEST_YEAR,
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
):
    if train_years is None or test_year is None:
        resolved_train_years, resolved_test_year = resolve_train_test_years(
            df,
            config=config,
        )
        train_years = resolved_train_years if train_years is None else train_years
        test_year = resolved_test_year if test_year is None else test_year

    train_df = df[df[COL_YEAR].isin(train_years)]
    test_df = df[df[COL_YEAR] == test_year]

    if train_df.empty:
        raise ValueError(f"{list(train_years)} 학습 데이터가 없습니다.")

    if test_df.empty:
        raise ValueError(f"{test_year} 테스트 데이터가 없습니다.")

    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS + [
            column for column in ENGINEERED_FEATURE_COLUMNS if column in df.columns
        ]
    missing_columns = [
        col for col in feature_columns + [target_column] if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    y = df[target_column]
    encoded_columns = fit_feature_encoder(train_df[feature_columns])
    X_train = transform_features(train_df[feature_columns], encoded_columns)
    y_train = y.loc[train_df.index]
    X_test = transform_features(test_df[feature_columns], encoded_columns)
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
        train = years before test_year
        test  = 2024
    """

    train_df = df[df[COL_YEAR] < test_year]
    test_df = df[df[COL_YEAR] == test_year]

    if train_df.empty:
        raise ValueError(f"{test_year} 이전 학습 데이터가 없습니다.")

    if test_df.empty:
        raise ValueError(f"{test_year} 테스트 데이터가 없습니다.")

    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS + [
            column for column in ENGINEERED_FEATURE_COLUMNS if column in df.columns
        ]
    missing_columns = [
        col for col in feature_columns + [target_column] if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    y = df[target_column]
    encoded_columns = fit_feature_encoder(train_df[feature_columns])
    X_train = transform_features(train_df[feature_columns], encoded_columns)
    y_train = y.loc[train_df.index]
    X_test = transform_features(test_df[feature_columns], encoded_columns)
    y_test = y.loc[test_df.index]

    return X_train, X_test, y_train, y_test
