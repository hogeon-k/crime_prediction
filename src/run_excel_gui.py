import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from gui.excel_window import ExcelWindow

if __name__ == "__main__":
    ExcelWindow().run()
