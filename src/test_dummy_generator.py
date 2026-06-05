"""
Test auto-generated crime data from GUI-style input dict.

Run:
    python test_dummy_generator.py
"""

import sys, os
from pathlib import Path

from model.excel_model import ProcessStatus
from services.dummy_generator import (
    DataExporter,
    DataGenerator,
    GenerationParams,
    get_user_input_gui,
    run_generation_pipeline,
)
from viewmodel.crime_viewmodel import CrimeViewModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


COL_CRIME = "\ubc94\uc8c4_\uc720\ud615"
COL_REGION = "\uc9c0\uc5ed"
COL_YEAR = "\uc5f0\ub3c4"
COL_INCIDENTS = "\ubc1c\uc0dd_\uac74\uc218"
COL_POP = "\uc778\uad6c\uc218"
COL_RATE = "\ubc94\uc8c4\uc728"

STEP_VALIDATE = "\uac80\uc99d"
STEP_MISSING = "\uacb0\uce21\uce58 \ucc98\ub9ac"
STEP_CONVERT = "\ud0c0\uc785 \ubcc0\ud658"

# Same shape as main_window._on_generate() raw dict
GUI_INPUT = {
    "data_count": 500,
    "year_start": 2022,
    "year_end": 2024,
    "region": [
        "\uc11c\uc6b8",
        "\ubd80\uc0b0",
        "\uc778\ucc9c",
        "\ub300\uad6c",
        "\ub300\uc804",
    ],
}


def test_params_conversion() -> GenerationParams:
    print("===== 1. GUI input -> GenerationParams =====")
    params = get_user_input_gui(GUI_INPUT)
    assert params.validate(), "GenerationParams.validate() failed"
    print(f"  data_count : {params.data_count}")
    print(f"  year_start : {params.year_start}")
    print(f"  year_end   : {params.year_end}")
    print(f"  region     : {params.region}")
    print("  PASS")
    return params


def test_raw_generation(params: GenerationParams) -> None:
    print("\n===== 2. Raw data generation =====")
    gen = DataGenerator(params, seed=42)
    df = gen.generate()

    assert len(df) == params.data_count
    assert list(df.columns) == [
        COL_CRIME,
        COL_REGION,
        COL_YEAR,
        COL_INCIDENTS,
        COL_POP,
    ]
    assert (df[COL_INCIDENTS] >= 0).all()
    assert (df[COL_POP] > 0).all()
    assert df[COL_REGION].isin(params.region).all()
    assert df[COL_YEAR].between(params.year_start, params.year_end).all()

    print(f"  shape      : {df.shape}")
    print(f"  year range : {df[COL_YEAR].min()} ~ {df[COL_YEAR].max()}")
    print(f"  regions    : {df[COL_REGION].nunique()}")
    print(f"  crime types: {df[COL_CRIME].nunique()}")
    print("  PASS")


def test_pipeline(params: GenerationParams) -> None:
    print("\n===== 3. Post-process pipeline =====")
    gen = DataGenerator(params, seed=42)
    df_raw = gen.generate()

    vm = CrimeViewModel(callback=lambda _s: None)
    vm.process_from_df(df_raw)

    assert vm.state.status == ProcessStatus.SUCCESS, vm.state.error_message
    assert vm.state.completed_steps == [STEP_VALIDATE, STEP_MISSING, STEP_CONVERT]

    df = vm.state.final_data
    assert df is not None
    assert COL_RATE in df.columns

    print(f"  steps done : {vm.state.completed_steps}")
    print(f"  avg rate   : {df[COL_RATE].mean():.2f}")
    print("  PASS")


def test_run_generation_pipeline() -> None:
    print("\n===== 4. run_generation_pipeline() =====")
    df = run_generation_pipeline(GUI_INPUT, seed=42)

    checks = {
        "row count": len(df) == GUI_INPUT["data_count"],
        "crime rate col": COL_RATE in df.columns,
        "no negative rate": (df[COL_RATE] >= 0).all(),
        "no nan rate": df[COL_RATE].isna().sum() == 0,
        "year range": df[COL_YEAR]
        .between(GUI_INPUT["year_start"], GUI_INPUT["year_end"])
        .all(),
        "valid regions": df[COL_REGION].isin(GUI_INPUT["region"]).all(),
        "positive pop": (df[COL_POP] > 0).all(),
    }

    all_passed = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    assert all_passed, "some checks failed"
    print("\n  head(5):")
    print(df.head().to_string(index=False))


def test_csv_export() -> None:
    print("\n===== 5. CSV export =====")
    df = run_generation_pipeline(GUI_INPUT, seed=42)
    out_path = Path(__file__).parent / "data" / "generated_crime_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = DataExporter.save_to_csv(df, str(out_path))
    assert ok, "CSV save failed"
    assert out_path.exists()
    print(f"  saved: {out_path}")
    print("  PASS")


def main() -> None:
    params = test_params_conversion()
    test_raw_generation(params)
    test_pipeline(params)
    test_run_generation_pipeline()
    test_csv_export()
    print("\nAll tests passed")


if __name__ == "__main__":
    main()
