import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from gui.crime_generator_window import CrimeGeneratorWindow

if __name__ == "__main__":
    CrimeGeneratorWindow().run()
