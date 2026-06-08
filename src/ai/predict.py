import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
PREDICTED_INCIDENTS_COLUMN = "예측_발생_건수"
PREDICTED_RATE_COLUMN = "예측_범죄율"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai.preprocessing import (
    DEFAULT_FEATURE_COLUMNS,
    encode_features,
    normalize_prediction_features,
)


def load_best_model(model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"저장된 모델 파일이 없습니다: {model_path}. "
            "먼저 `python src/ai/train.py`를 실행해 best_model.pkl을 생성하세요."
        )

    with model_path.open("rb") as file:
        return pickle.load(file)


def prepare_prediction_features(model, X):
    if not isinstance(X, pd.DataFrame):
        return X

    encoded_columns = getattr(model, "feature_columns", None)

    if set(DEFAULT_FEATURE_COLUMNS).issubset(X.columns):
        X = X[DEFAULT_FEATURE_COLUMNS]
        X = normalize_prediction_features(X, encoded_columns=encoded_columns)

    return encode_features(X, encoded_columns=encoded_columns)


def _print_input_diversity(raw_X, debug_printer=print):
    if not isinstance(raw_X, pd.DataFrame) or not set(DEFAULT_FEATURE_COLUMNS).issubset(raw_X.columns):
        return

    debug_printer("\n========== 원본 입력 다양성 ==========")
    debug_printer(f"연도 unique 개수: {raw_X['연도'].nunique(dropna=False)}")
    debug_printer(f"지역 unique 개수: {raw_X['지역'].nunique(dropna=False)}")
    debug_printer(f"범죄_유형 unique 개수: {raw_X['범죄_유형'].nunique(dropna=False)}")
    population = pd.to_numeric(raw_X["인구수"], errors="coerce")
    debug_printer(f"인구수 min: {population.min()}")
    debug_printer(f"인구수 max: {population.max()}")
    debug_printer(f"인구수 nunique: {population.nunique(dropna=False)}")


def _feature_importance_from_tree_splits(model, final_X):
    if not isinstance(final_X, pd.DataFrame) or not hasattr(model, "trees"):
        return None

    feature_counts = Counter()

    def visit(node):
        if node is None or getattr(node, "value", None) is not None:
            return
        feature_index = getattr(node, "feature_index", None)
        if feature_index is not None and 0 <= feature_index < len(final_X.columns):
            feature_counts[final_X.columns[feature_index]] += 1
        visit(getattr(node, "left", None))
        visit(getattr(node, "right", None))

    for tree in getattr(model, "trees", []):
        visit(getattr(tree, "root", None))

    if not feature_counts:
        return pd.Series(dtype=float)

    total = sum(feature_counts.values())
    return pd.Series(
        {feature: count / total for feature, count in feature_counts.items()}
    ).sort_values(ascending=False)


def _feature_importance(model, final_X):
    if hasattr(model, "feature_importances_") and isinstance(final_X, pd.DataFrame):
        return pd.Series(
            model.feature_importances_,
            index=final_X.columns,
            dtype=float,
        ).sort_values(ascending=False)

    return _feature_importance_from_tree_splits(model, final_X)


def _print_feature_importance_debug(model, final_X, debug_printer=print):
    importance = _feature_importance(model, final_X)
    if importance is None:
        debug_printer("\n========== Feature Importance ==========")
        debug_printer("지원하지 않는 모델 타입이라 feature importance를 계산하지 못했습니다.")
        return

    debug_printer("\n========== Feature Importance 상위 10개 ==========")
    if importance.empty:
        debug_printer("트리 split 정보가 없어 feature importance가 비어 있습니다.")
        return

    debug_printer(importance.head(10).to_string())

    for feature in ("인구수", "연도"):
        value = float(importance.get(feature, 0.0))
        if value <= 0:
            debug_printer(
                f"경고: 모델 split 기준으로 '{feature}' 영향이 거의 없거나 없습니다. "
                "현재 feature만으로는 예측값이 비슷하게 나올 수 있습니다."
            )


def _print_prediction_distribution(predictions, debug_printer=print):
    if predictions is None:
        return

    series = pd.Series(predictions, dtype=float)
    debug_printer("\n========== 예측 결과 분포 ==========")
    debug_printer(f"예측_발생_건수 min: {series.min()}")
    debug_printer(f"예측_발생_건수 max: {series.max()}")
    debug_printer(f"예측_발생_건수 mean: {series.mean()}")
    debug_printer(f"예측_발생_건수 std: {series.std(ddof=0)}")
    debug_printer(f"예측_발생_건수 unique 개수: {series.nunique(dropna=False)}")


def _print_similar_prediction_groups(raw_X, predictions, debug_printer=print):
    if predictions is None or not isinstance(raw_X, pd.DataFrame):
        return

    debug_printer("\n========== 입력은 다른데 예측이 거의 같은 행 ==========")
    comparable = raw_X.copy()
    comparable[PREDICTED_INCIDENTS_COLUMN] = pd.Series(predictions, index=comparable.index, dtype=float)
    comparable["_예측_round_2"] = comparable[PREDICTED_INCIDENTS_COLUMN].round(2)

    groups_found = False
    for rounded_value, group in comparable.groupby("_예측_round_2", dropna=False):
        if len(group) < 2:
            continue

        input_unique_rows = group[DEFAULT_FEATURE_COLUMNS].drop_duplicates()
        if len(input_unique_rows) < 2:
            continue

        groups_found = True
        debug_printer(f"\n예측_발생_건수 ~= {rounded_value} 그룹 ({len(group)}행)")
        debug_printer(
            group[DEFAULT_FEATURE_COLUMNS + [PREDICTED_INCIDENTS_COLUMN]]
            .head(10)
            .to_string(index=False)
        )

    if not groups_found:
        debug_printer("소수점 둘째 자리 기준으로 같은 예측값을 가진 서로 다른 입력 그룹이 없습니다.")


def _print_prediction_debug(model, raw_X, final_X, predictions=None, debug_printer=print):
    debug_printer("\n========== 예측 입력 원본 ==========")
    if isinstance(raw_X, pd.DataFrame):
        debug_printer(f"columns: {list(raw_X.columns)}")
        if set(DEFAULT_FEATURE_COLUMNS).issubset(raw_X.columns):
            debug_printer(raw_X[DEFAULT_FEATURE_COLUMNS].head(10).to_string(index=False))
        else:
            debug_printer(raw_X.head(10).to_string(index=False))
    else:
        debug_printer(f"type: {type(raw_X).__name__}")

    _print_input_diversity(raw_X, debug_printer=debug_printer)

    feature_columns = getattr(model, "feature_columns", None)
    debug_printer("\n========== 저장된 model.feature_columns ==========")
    debug_printer(f"exists: {feature_columns is not None}")
    debug_printer(f"count: {len(feature_columns) if feature_columns is not None else 0}")
    debug_printer(feature_columns)

    debug_printer("\n========== model.predict() 직전 최종 feature ==========")
    if isinstance(final_X, pd.DataFrame):
        debug_printer(f"shape: {final_X.shape}")
        debug_printer(f"columns: {list(final_X.columns)}")
        debug_printer(final_X.head(10).to_string(index=False))
        debug_printer("\n각 컬럼별 nunique():")
        nunique = final_X.nunique()
        debug_printer(nunique.to_string())
        debug_printer(f"\nfinal X unique rows: {len(final_X.drop_duplicates())}/{len(final_X)}")
        debug_printer(f"final X duplicated rows: {final_X.duplicated(keep=False).tolist()}")
        changing_columns = nunique[nunique > 1].index.tolist()
        zero_dummy_columns = [
            column for column in final_X.columns
            if column not in ("연도", "인구수") and (final_X[column] == 0).all()
        ]
        constant_columns = nunique[nunique <= 1].index.tolist()
        debug_printer("\n========== encoded feature 변화 요약 ==========")
        debug_printer(f"nunique() > 1 컬럼 목록 ({len(changing_columns)}개): {changing_columns}")
        debug_printer(f"모든 행에서 0인 dummy 컬럼 개수: {len(zero_dummy_columns)}")
        debug_printer(f"모든 행에서 같은 값인 컬럼 개수: {len(constant_columns)}")
    else:
        debug_printer(f"type: {type(final_X).__name__}")

    _print_prediction_distribution(predictions, debug_printer=debug_printer)
    _print_similar_prediction_groups(raw_X, predictions, debug_printer=debug_printer)
    _print_feature_importance_debug(model, final_X, debug_printer=debug_printer)


def _clip_negative_predictions(predictions):
    return [max(0.0, float(value)) for value in predictions]


def predict(*args, model_path=DEFAULT_MODEL_PATH, debug=False, debug_printer=print):
    if len(args) == 1:
        model = load_best_model(model_path)
        X = args[0]
    elif len(args) == 2:
        model, X = args
    else:
        raise TypeError("predict()는 predict(X) 또는 predict(model, X) 형태로 호출하세요.")

    if model is None:
        raise ValueError("예측할 모델이 없습니다.")

    if not hasattr(model, "predict"):
        raise TypeError("model 객체에 predict() 메서드가 없습니다.")

    final_X = prepare_prediction_features(model, X)
    predictions = model.predict(final_X)
    clipped_predictions = _clip_negative_predictions(predictions)
    if debug:
        _print_prediction_debug(
            model,
            X,
            final_X,
            predictions=clipped_predictions,
            debug_printer=debug_printer,
        )

    return clipped_predictions


def predict_one(
    year: int,
    region: str,
    crime_type: str,
    population: int,
    model=None,
    model_path=DEFAULT_MODEL_PATH,
):
    if model is None:
        model = load_best_model(model_path)

    X = pd.DataFrame(
        {
            "연도": [year],
            "지역": [region],
            "범죄_유형": [crime_type],
            "인구수": [population],
        }
    )

    prediction = predict(model, X)

    return float(prediction[0])


def validate_prediction_input(df):
    missing_columns = [column for column in DEFAULT_FEATURE_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(f"예측에 필요한 컬럼이 없습니다: {missing_columns}")


def predict_from_dataframe(df, debug=False, debug_printer=print):
    validate_prediction_input(df)

    result_df = df.copy()
    predicted_incidents = predict(
        result_df[DEFAULT_FEATURE_COLUMNS],
        debug=debug,
        debug_printer=debug_printer,
    )
    result_df[PREDICTED_INCIDENTS_COLUMN] = predicted_incidents
    result_df[PREDICTED_RATE_COLUMN] = (
        result_df[PREDICTED_INCIDENTS_COLUMN] / result_df["인구수"] * 100000
    )

    return result_df


def _read_prediction_file(input_path):
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path)

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(input_path)

    raise ValueError("지원하지 않는 입력 파일 형식입니다. csv, xlsx, xls 파일만 사용할 수 있습니다.")


def _write_prediction_file(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return

    if suffix in (".xlsx", ".xls"):
        df.to_excel(output_path, index=False, engine="openpyxl")
        return

    raise ValueError("지원하지 않는 출력 파일 형식입니다. csv, xlsx, xls 파일만 사용할 수 있습니다.")


def predict_from_file(input_path, output_path, debug=False, debug_printer=print):
    df = _read_prediction_file(input_path)
    result_df = predict_from_dataframe(df, debug=debug, debug_printer=debug_printer)
    _write_prediction_file(result_df, output_path)

    return result_df
