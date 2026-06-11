from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai.preprocessing import (  # noqa: E402
    excluded_previous_feature_years,
    feature_available_years,
    resolve_train_test_years,
)
from ai.train import get_default_government_files, make_training_dataframe  # noqa: E402
from constants import COL_CRIME_TYPE, COL_REGION, COL_YEAR  # noqa: E402
from services.crime_service import CrimeService, normalize_crime_type  # noqa: E402


def _read_csv_any_encoding(path: str | Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"CSV 파일을 읽을 수 없습니다: {path}") from last_error


def _raw_crime_type_values(crime_files: list[str]) -> pd.Series:
    values = []
    for path in crime_files:
        raw = _read_csv_any_encoding(path)
        if "범죄중분류" in raw.columns:
            values.extend(raw["범죄중분류"].dropna().astype(str).tolist())
    return pd.Series(values, dtype=str)


REPORT_VALUES = {
    "total_rows": 4522,
    "region_count": 17,
    "yearly_crime_type_count": 38,
    "reported_crime_type_count": 38,
    "merge_failed_count": 0,
    "holdout_train_years": [2019, 2020, 2021, 2022, 2023],
    "holdout_test_year": 2024,
    "train_row_count": 3230,
    "test_row_count": 646,
    "linear_test_r2": 0.9518,
    "linear_test_rmse": 1788.76,
    "linear_test_mae": 700.13,
    "walk_forward_linear_mean_r2": 0.9656,
    "walk_forward_linear_mean_rmse": 1340.21,
    "walk_forward_linear_mean_mae": 535.81,
    "final_model": "linear",
}


def _status(name: str, report, actual, tolerance: float = 0.0001) -> str:
    if isinstance(report, float) or isinstance(actual, float):
        ok = math.isclose(float(report), float(actual), rel_tol=tolerance, abs_tol=tolerance)
    else:
        ok = report == actual
    label = "OK" if ok else "MISMATCH"
    return f"[{label}] {name}: report={report}, actual={actual}"


def _load_model_info() -> dict:
    path = ROOT_DIR / "models" / "model_info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    crime_files, pop_files = get_default_government_files()
    service = CrimeService()
    raw_crime = service._load_crime(crime_files)
    df = make_training_dataframe(crime_files, pop_files)
    report = df.attrs.get("preprocessing_report", {})
    model_info = _load_model_info()
    metrics = model_info.get("metrics", {})
    wf = model_info.get("walk_forward_averages", {})
    linear_wf = wf.get("linear", {})

    train_years, test_year = resolve_train_test_years(df)
    actual_values = {
        "total_rows": len(df),
        "region_count": int(df[COL_REGION].nunique()),
        "yearly_crime_type_count": int(df.groupby(COL_YEAR)[COL_CRIME_TYPE].nunique().max()),
        "reported_crime_type_count": int(df[COL_CRIME_TYPE].nunique()),
        "merge_failed_count": int(report.get("merge_failed_key_count", 0)),
        "holdout_train_years": list(train_years),
        "holdout_test_year": int(test_year),
        "train_row_count": int(df[df[COL_YEAR].isin(train_years)].shape[0]),
        "test_row_count": int(df[df[COL_YEAR] == test_year].shape[0]),
        "linear_test_r2": metrics.get("r2"),
        "linear_test_rmse": metrics.get("rmse"),
        "linear_test_mae": metrics.get("mae"),
        "walk_forward_linear_mean_r2": linear_wf.get("mean_r2"),
        "walk_forward_linear_mean_rmse": linear_wf.get("mean_rmse"),
        "walk_forward_linear_mean_mae": linear_wf.get("mean_mae"),
        "final_model": model_info.get("best_model"),
    }

    print("## Report consistency check")
    for key, expected in REPORT_VALUES.items():
        actual = actual_values.get(key)
        if actual is None:
            print(f"[MISSING] {key}: report={expected}, actual=None")
        else:
            print(_status(key, expected, actual, tolerance=0.001))

    raw_crime_type_values = _raw_crime_type_values(crime_files)
    before_norm = raw_crime_type_values.nunique()
    after_norm = raw_crime_type_values.map(normalize_crime_type).nunique()
    model_crime_one_hot_count = sum(
        1
        for column in model_info.get("feature_columns", [])
        if str(column).startswith(f"{COL_CRIME_TYPE}_")
    )

    print("\n## Crime type diagnostics")
    print(f"raw_crime_type_unique_before_normalization={before_norm}")
    print(f"raw_crime_type_unique_after_normalization={after_norm}")
    print(f"merged_crime_type_unique={df[COL_CRIME_TYPE].nunique()}")
    print(f"yearly_crime_type_counts={df.groupby(COL_YEAR)[COL_CRIME_TYPE].nunique().to_dict()}")
    print(f"model_crime_type_one_hot_count={model_crime_one_hot_count}")
    print(f"feature_available_years={feature_available_years(df)}")
    print(f"excluded_years={excluded_previous_feature_years(df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
