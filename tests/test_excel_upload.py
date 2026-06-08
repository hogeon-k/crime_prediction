"""
Excel/CSV upload pipeline test (no GUI).

Run:
    python test_excel_upload.py
"""

from pathlib import Path

import pandas as pd

import _path_setup  # pylint: disable=unused-import
from model.excel_model import UploadParams
from services.crime_service import CrimeService, _normalize_crime_region_to_sido
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


def test_government_pipeline_if_data_exists() -> None:
    print("\n===== 4. run_excel_pipeline (government) =====")
    crime_files = sorted(DATA_DIR.glob("crime_region_20*.csv"))
    pop_files = sorted(DATA_DIR.glob("pop_20*.csv"))
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
