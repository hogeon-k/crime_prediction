"""
MVVM 패턴 실행 진입점

구조:
    Model       : model/excel_model.py      (CrimeState, ProcessResult, ValidationRule)
    Service     : services/crime_service.py (비즈니스 로직 - 데이터 로드·검증·변환)
    ViewModel   : crime_viewmodel.py        (파이프라인 조율, 상태 관리)
    View        : crime_view.py             (상태를 받아 출력)

파일 규칙 (data/ 폴더에 넣기만 하면 자동 인식):
    범죄 데이터 : crime_region_20XX.csv
    인구 데이터 : 2015-2025_pop.csv 같은 통합 파일 또는 pop_20XX.csv
"""

import re
import sys
from glob import glob
from pathlib import Path

from view.crime_view import CrimeView
from viewmodel.crime_viewmodel import CrimeViewModel

DATA_DIR = Path(__file__).parent / "data"


def collect_files() -> tuple[list[str], list[str]]:
    """
    data/ 폴더에서 파일을 자동 수집합니다.
    - 코드 수정 없이 파일만 추가하면 자동으로 인식됩니다.
    - 연도 오름차순 정렬 (파이프라인 내부에서 연도 컬럼으로 구분)
    """
    crime_files = sorted(glob(str(DATA_DIR / "crime_region_20*.csv")))
    wide_pop_files = sorted(DATA_DIR.glob("*pop*.csv"))
    wide_pop_files = [
        path
        for path in wide_pop_files
        if not path.name.startswith("pop_20") and re.search(r"20\d{2}.*20\d{2}", path.stem)
    ]
    pop_files = [str(path) for path in wide_pop_files] or sorted(glob(str(DATA_DIR / "pop_20*.csv")))
    return crime_files, pop_files


def validate_files(crime_files: list[str], pop_files: list[str]) -> bool:
    """수집된 파일이 실행 가능한 상태인지 사전 점검합니다."""
    ok = True

    if not crime_files:
        print(f"❌ 범죄 데이터 파일 없음 (crime_region_20XX.csv → {DATA_DIR})")
        ok = False
    if not pop_files:
        print(f"❌ 인구 데이터 파일 없음 (pop_20XX.csv → {DATA_DIR})")
        ok = False

    if ok:
        print(
            f"📂 범죄 파일 {len(crime_files)}개: {[Path(f).name for f in crime_files]}"
        )
        print(f"📂 인구 파일 {len(pop_files)}개: {[Path(f).name for f in pop_files]}")

    return ok


def main() -> None:
    crime_files, pop_files = collect_files()

    if not validate_files(crime_files, pop_files):
        sys.exit(1)

    view = CrimeView()
    vm = CrimeViewModel(callback=view.render)

    vm.process(
        crime_files=crime_files,
        pop_files=pop_files,
    )


if __name__ == "__main__":
    main()
