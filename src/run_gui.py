"""Deprecated legacy GUI entry point.

Use run_app.py as the final application entry point. This script is kept for
manual fallback execution of the old sample-data window.
"""

from gui.crime_generator_window import CrimeGeneratorWindow

if __name__ == "__main__":
    CrimeGeneratorWindow().run()
