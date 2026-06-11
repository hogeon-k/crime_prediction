"""
Excel/CSV upload pipeline test (no GUI).

Run:
    python test_excel_upload.py
"""

from pathlib import Path

import pandas as pd
import pytest

import _path_setup  # pylint: disable=unused-import
from ai.train import get_default_government_files
from model.excel_model import UploadParams
from services.crime_service import (
    CrimeService,
    _normalize_crime_region_to_sido,
    normalize_crime_type,
)
from services.excel_pipeline import run_excel_pipeline


COL_CRIME = "\ubc94\uc8c4_\uc720\ud615"
COL_REGION = "\uc9c0\uc5ed"
COL_YEAR = "\uc5f0\ub3c4"
COL_INCIDENTS = "\ubc1c\uc0dd_\uac74\uc218"
COL_POP = "\uc778\uad6c\uc218"
COL_RATE = "\ubc94\uc8c4\uc728"

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_XLSX = DATA_DIR / "sample_upload.xlsx"


def _make_sample_xlsx() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            COL_CRIME: ["\uc808\ub3c4", "\ud3ed\ub825", "\uc0ac\uae30"],
            COL_REGION: ["\uc11c\uc6b8", "\ubd80\uc0b0", "\uc778\ucc9c"],
            COL_YEAR: [2022, 2023, 2024],
            COL_INCIDENTS: [120, 90, 200],
            COL_POP: [9_500_000, 3_400_000, 3_000_000],
        }
    )
    df.to_excel(SAMPLE_XLSX, index=False, engine="openpyxl")
    return SAMPLE_XLSX


def test_load_uploaded() -> None:
    print("===== 1. load_uploaded (xlsx) =====")
    path = _make_sample_xlsx()
    service = CrimeService()
    result = service.load_uploaded(str(path))
    assert result.success, result.message
    assert result.data is not None
    assert len(result.data) == 3
    print(f"  rows: {len(result.data)}")
    print("  PASS")


def test_load_uploaded_requires_population(tmp_path) -> None:
    path = tmp_path / "missing_population.xlsx"
    df = pd.DataFrame(
        {
            COL_CRIME: ["절도"],
            COL_REGION: ["서울"],
            COL_YEAR: [2024],
            COL_INCIDENTS: [120],
        }
    )
    df.to_excel(path, index=False, engine="openpyxl")

    result = CrimeService().load_uploaded(str(path))

    assert not result.success
    assert COL_POP in result.message

    params = UploadParams(mode="standard", standard_file=str(path))
    with pytest.raises(ValueError, match=COL_POP):
        run_excel_pipeline(params)


def test_standard_pipeline() -> None:
    print("\n===== 2. run_excel_pipeline (standard) =====")
    path = _make_sample_xlsx()
    params = UploadParams(mode="standard", standard_file=str(path))
    df = run_excel_pipeline(params)
    assert COL_RATE in df.columns
    assert len(df) == 3
    assert (df[COL_RATE] >= 0).all()
    print(f"  columns: {list(df.columns)}")
    print("  PASS")


def test_government_region_normalization() -> None:
    print("\n===== 3. government region normalization =====")
    regions = pd.Series(["서울종로구", "서울 종로구", "서울", "경기 고양"])
    normalized = _normalize_crime_region_to_sido(regions)
    assert normalized.tolist() == ["서울", "서울", "서울", "경기"]
    print("  PASS")


def test_crime_type_spacing_and_separators_are_normalized() -> None:
    values = [
        "기타 강간 강제추행등",
        "기타 강간/강제추행등",
        "기타강간강제추행등",
        "문서 인장",
        "문서/인장",
        "문서인장",
        "체포/감금",
        "체포감금",
    ]

    assert [normalize_crime_type(value) for value in values] == [
        "기타강간강제추행등",
        "기타강간강제추행등",
        "기타강간강제추행등",
        "문서인장",
        "문서인장",
        "문서인장",
        "체포감금",
        "체포감금",
    ]


def test_government_merge_fails_when_population_missing(tmp_path) -> None:
    crime_path = tmp_path / "crime_region_2024.csv"
    pop_path = tmp_path / "pop_2024.csv"
    pd.DataFrame(
        {
            "범죄대분류": ["재산범죄"],
            "범죄중분류": ["절도범죄"],
            "서울": [120],
        }
    ).to_csv(crime_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "시도명": ["부산광역시"],
            "통계년월": [202401],
            "계": [3_300_000],
        }
    ).to_csv(pop_path, index=False, encoding="utf-8-sig")

    result = CrimeService().load_and_merge([str(crime_path)], [str(pop_path)])

    assert not result.success
    assert "인구수 결측" in result.message


def test_government_pipeline_if_data_exists() -> None:
    print("\n===== 4. run_excel_pipeline (government) =====")
    crime_files, pop_files = get_default_government_files()
    if not crime_files or not pop_files:
        print("  SKIP (no government csv in data/)")
        return

    params = UploadParams(
        mode="government",
        crime_files=[str(f) for f in crime_files],
        pop_files=[str(f) for f in pop_files],
    )
    df = run_excel_pipeline(params)
    assert COL_RATE in df.columns
    assert len(df) > 0
    print(f"  rows: {len(df):,}")
    print("  PASS")


def main() -> None:
    test_load_uploaded()
    test_standard_pipeline()
    test_government_region_normalization()
    test_government_pipeline_if_data_exists()
    print("\nAll tests passed")


if __name__ == "__main__":
    main()
