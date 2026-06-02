from pathlib import Path

from view.crime_view import CrimeView
from viewmodel.crime_viewmodel import CrimeViewModel


def main():

    crime_files = sorted(str(p) for p in Path("data").glob("crime_region_*.csv"))

    pop_files = sorted(str(p) for p in Path("data").glob("pop_*.csv"))

    view = CrimeView()

    vm = CrimeViewModel(callback=view.render)

    vm.process(
        crime_files,
        pop_files,
    )


if __name__ == "__main__":
    main()
