"""Deprecated legacy Excel GUI entry point.

Use run_app.py as the final application entry point. This script is kept for
manual fallback execution of the old Excel window.
"""

from gui.excel_window import ExcelWindow

if __name__ == "__main__":
    ExcelWindow().run()
