from pathlib import Path

import _path_setup  # pylint: disable=unused-import
from services.crime_service import CrimeService


DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "data"


def test_government_data_pipeline() -> None:
    crime_files = sorted(DATA_DIR.glob("crime_region_20*.csv"))
    pop_files = sorted(DATA_DIR.glob("pop_20*.csv"))

    if not crime_files or not pop_files:
        print("SKIP: no government csv files in src/data")
        return

    service = CrimeService()

    result = service.load_and_merge(
        [str(path) for path in crime_files],
        [str(path) for path in pop_files],
    )
    assert result.success, result.message
    assert result.data is not None

    result = service.validate(result.data)
    assert result.success, result.message
    assert result.data is not None

    result = service.handle_missing(result.data)
    assert result.success, result.message
    assert result.data is not None

    result = service.convert_types(result.data)
    assert result.success, result.message
    assert result.data is not None


if __name__ == "__main__":
    test_government_data_pipeline()
    print("All tests passed")
