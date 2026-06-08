from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai.predict import (  # noqa: E402
    PREDICTED_INCIDENTS_COLUMN,
    PREDICTED_RATE_COLUMN,
    predict_from_file,
    predict_one,
)
from gui.crime_generator_window import (  # noqa: E402
    ACCENT,
    BG,
    BORDER,
    DANGER,
    FONT_BOLD,
    FONT_LABEL,
    FONT_MONO,
    FONT_SMALL,
    FONT_TITLE,
    PANEL_BG,
    SUCCESS,
    TEXT_PRI,
    TEXT_SEC,
    _btn,
    _card,
    _label,
)
from gui.excel_window import ExcelWindow  # noqa: E402
from model.excel_model import UploadParams  # noqa: E402
from services.dummy_generator import (  # noqa: E402
    VALID_REGIONS,
    DataExporter,
    run_generation_pipeline,
)
from services.excel_pipeline import run_excel_pipeline  # noqa: E402

PREDICTION_OUTPUT_PATH = ROOT_DIR / "data" / "prediction_result.xlsx"
DEFAULT_GENERATED_PATH = SRC_DIR / "data" / "generated_crime_data.csv"
DEFAULT_PROCESSED_PATH = SRC_DIR / "data" / "processed_crime_data.csv"

COL_CRIME = "범죄_유형"
COL_REGION = "지역"
COL_YEAR = "연도"
COL_INCIDENTS = "발생_건수"
COL_POP = "인구수"
COL_RATE = "범죄율"


def friendly_prediction_error(message: str) -> str:
    return ExcelWindow._format_prediction_error(message)


class TextLogger:
    def __init__(self, root: tk.Tk, text: tk.Text) -> None:
        self.root = root
        self.text = text

    def write(self, message: str) -> None:
        self.root.after(0, self._append, message)

    def _append(self, message: str) -> None:
        self.text.insert("end", message)
        if not message.endswith("\n"):
            self.text.insert("end", "\n")
        self.text.see("end")


class BaseTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, app: "CrimePredictionApp") -> None:
        super().__init__(parent)
        self.app = app
        self.root = app.root
        self._busy = False
        self._buttons: list[tk.Button] = []
        self.configure(style="App.TFrame")

    def log(self, message: str) -> None:
        self.app.log(message)

    def add_button(self, button: tk.Button) -> tk.Button:
        self._buttons.append(button)
        return button

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for button in self._buttons:
            button.config(state=state)

    @staticmethod
    def format_number(value, digits: int | None = None) -> str:
        if value == "" or pd.isna(value):
            return ""
        if digits is None:
            return f"{float(value):,.0f}"
        return f"{float(value):,.{digits}f}"

    def make_table(self, parent: tk.Widget, columns: list[str]) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=100, anchor="center")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        return tree

    def clear_table(self, tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)


class DataGenerationTab(BaseTab):
    def __init__(self, parent: ttk.Notebook, app: "CrimePredictionApp") -> None:
        super().__init__(parent, app)
        self._df: pd.DataFrame | None = None
        self._build()

    def _build(self) -> None:
        left = tk.Frame(self, bg=BG, width=330)
        left.pack(side="left", fill="y", padx=12, pady=12)
        left.pack_propagate(False)
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)

        card = _card(left)
        card.pack(fill="both", expand=True)
        inner = tk.Frame(card, bg=PANEL_BG, padx=14, pady=14)
        inner.pack(fill="both", expand=True)

        _label(inner, "더미 데이터 생성", font=FONT_TITLE).pack(anchor="w", pady=(0, 10))
        self.count_var = tk.IntVar(value=500)
        self.year_start_var = tk.IntVar(value=2022)
        self.year_end_var = tk.IntVar(value=2024)
        self._spin_row(inner, "데이터 수", self.count_var, 1, 10000)
        self._spin_row(inner, "시작 연도", self.year_start_var, 2020, 2026)
        self._spin_row(inner, "종료 연도", self.year_end_var, 2020, 2026)

        _label(inner, "지역", fg=TEXT_SEC).pack(anchor="w", pady=(8, 4))
        self.region_lb = tk.Listbox(inner, selectmode="multiple", height=8, font=FONT_SMALL)
        self.region_lb.pack(fill="x")
        for region in VALID_REGIONS:
            self.region_lb.insert("end", region)
        for i in range(min(3, len(VALID_REGIONS))):
            self.region_lb.selection_set(i)

        _label(inner, "저장 경로", fg=TEXT_SEC).pack(anchor="w", pady=(12, 4))
        path_row = tk.Frame(inner, bg=PANEL_BG)
        path_row.pack(fill="x")
        self.path_var = tk.StringVar(value=str(DEFAULT_GENERATED_PATH))
        ttk.Entry(path_row, textvariable=self.path_var, font=FONT_SMALL).pack(
            side="left", fill="x", expand=True
        )
        tk.Button(path_row, text="...", command=self._browse_save, relief="flat", bg=BORDER).pack(side="right")

        self.add_button(_btn(inner, "생성", self._on_generate, bg=ACCENT)).pack(fill="x", pady=(14, 6))
        self.add_button(_btn(inner, "CSV 저장", self._on_save, bg=SUCCESS)).pack(fill="x")

        table_card = _card(right)
        table_card.pack(fill="both", expand=True)
        table_inner = tk.Frame(table_card, bg=PANEL_BG, padx=14, pady=10)
        table_inner.pack(fill="both", expand=True)
        _label(table_inner, "미리보기", font=FONT_BOLD).pack(anchor="w", pady=(0, 6))
        self.tree = self.make_table(
            table_inner, [COL_CRIME, COL_REGION, COL_YEAR, COL_INCIDENTS, COL_POP, COL_RATE]
        )

    def _spin_row(self, parent: tk.Widget, label: str, var: tk.IntVar, start: int, end: int) -> None:
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", pady=3)
        _label(row, label, fg=TEXT_SEC, width=10, anchor="w").pack(side="left")
        ttk.Spinbox(row, from_=start, to=end, textvariable=var, width=10).pack(side="left")

    def _browse_save(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=DEFAULT_GENERATED_PATH.name,
        )
        if path:
            self.path_var.set(path)

    def _on_generate(self) -> None:
        if self._busy:
            return
        regions = [self.region_lb.get(i) for i in self.region_lb.curselection()]
        raw = {
            "data_count": self.count_var.get(),
            "year_start": self.year_start_var.get(),
            "year_end": self.year_end_var.get(),
            "region": regions,
        }
        self.set_busy(True)
        self.log("더미 데이터 생성 시작")
        threading.Thread(target=self._run_generate, args=(raw,), daemon=True).start()

    def _run_generate(self, raw: dict) -> None:
        try:
            df = run_generation_pipeline(raw)
            self._df = df
            self.root.after(0, self._show_dataframe, df)
            self.log(f"더미 데이터 생성 완료: {len(df):,}행")
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("더미 데이터 생성 실패", str(exc)))
            self.log(f"더미 데이터 생성 실패: {exc}")
        finally:
            self.root.after(0, lambda: self.set_busy(False))

    def _show_dataframe(self, df: pd.DataFrame) -> None:
        self.clear_table(self.tree)
        for _, row in df.head(10).iterrows():
            self.tree.insert(
                "",
                "end",
                values=[
                    row.get(COL_CRIME, ""),
                    row.get(COL_REGION, ""),
                    row.get(COL_YEAR, ""),
                    self.format_number(row.get(COL_INCIDENTS, "")),
                    self.format_number(row.get(COL_POP, "")),
                    self.format_number(row.get(COL_RATE, ""), 2),
                ],
            )

    def _on_save(self) -> None:
        if self._df is None:
            messagebox.showwarning("저장 불가", "먼저 데이터를 생성하세요.")
            return
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("경로 없음", "저장 경로를 입력하세요.")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if DataExporter.save_to_csv(self._df, path):
            self.log(f"더미 CSV 저장 완료: {path}")
            messagebox.showinfo("저장 완료", path)
        else:
            messagebox.showerror("저장 실패", path)


class UploadPreprocessTab(BaseTab):
    def __init__(self, parent: ttk.Notebook, app: "CrimePredictionApp") -> None:
        super().__init__(parent, app)
        self._df: pd.DataFrame | None = None
        self.crime_files: list[str] = []
        self.pop_files: list[str] = []
        self._build()

    def _build(self) -> None:
        left = tk.Frame(self, bg=BG, width=360)
        left.pack(side="left", fill="y", padx=12, pady=12)
        left.pack_propagate(False)
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)

        card = _card(left)
        card.pack(fill="both", expand=True)
        inner = tk.Frame(card, bg=PANEL_BG, padx=14, pady=14)
        inner.pack(fill="both", expand=True)
        _label(inner, "Excel/CSV 업로드 및 전처리", font=FONT_TITLE).pack(anchor="w", pady=(0, 10))

        self.mode_var = tk.StringVar(value="standard")
        ttk.Radiobutton(inner, text="표준 양식", variable=self.mode_var, value="standard", command=self._sync_mode).pack(anchor="w")
        ttk.Radiobutton(inner, text="공공데이터", variable=self.mode_var, value="government", command=self._sync_mode).pack(anchor="w")

        self.standard_frame = tk.Frame(inner, bg=PANEL_BG)
        _label(self.standard_frame, "표준 파일", fg=TEXT_SEC).pack(anchor="w", pady=(8, 4))
        row = tk.Frame(self.standard_frame, bg=PANEL_BG)
        row.pack(fill="x")
        self.standard_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.standard_var, font=FONT_SMALL).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="...", command=self._browse_standard, relief="flat", bg=BORDER).pack(side="right")

        self.gov_frame = tk.Frame(inner, bg=PANEL_BG)
        self.crime_list = self._file_list(self.gov_frame, "범죄 파일", self._browse_crime)
        self.pop_list = self._file_list(self.gov_frame, "인구 파일", self._browse_pop)

        _label(inner, "저장 경로", fg=TEXT_SEC).pack(anchor="w", pady=(12, 4))
        path_row = tk.Frame(inner, bg=PANEL_BG)
        path_row.pack(fill="x")
        self.path_var = tk.StringVar(value=str(DEFAULT_PROCESSED_PATH))
        ttk.Entry(path_row, textvariable=self.path_var, font=FONT_SMALL).pack(side="left", fill="x", expand=True)
        tk.Button(path_row, text="...", command=self._browse_save, relief="flat", bg=BORDER).pack(side="right")

        self.add_button(_btn(inner, "업로드 & 전처리", self._on_process, bg=ACCENT)).pack(fill="x", pady=(14, 6))
        self.add_button(_btn(inner, "CSV 저장", self._on_save, bg=SUCCESS)).pack(fill="x")
        self._sync_mode()

        table_card = _card(right)
        table_card.pack(fill="both", expand=True)
        table_inner = tk.Frame(table_card, bg=PANEL_BG, padx=14, pady=10)
        table_inner.pack(fill="both", expand=True)
        _label(table_inner, "처리 결과 미리보기", font=FONT_BOLD).pack(anchor="w", pady=(0, 6))
        self.tree = self.make_table(
            table_inner, [COL_CRIME, COL_REGION, COL_YEAR, COL_INCIDENTS, COL_POP, COL_RATE]
        )

    def _file_list(self, parent: tk.Widget, label: str, command) -> tk.Listbox:
        _label(parent, label, fg=TEXT_SEC).pack(anchor="w", pady=(8, 4))
        lb = tk.Listbox(parent, height=4, font=FONT_SMALL)
        lb.pack(fill="x")
        tk.Button(parent, text="추가", command=command, relief="flat", bg=BORDER).pack(anchor="w", pady=(3, 0))
        return lb

    def _sync_mode(self) -> None:
        if self.mode_var.get() == "standard":
            self.gov_frame.pack_forget()
            self.standard_frame.pack(fill="x", pady=8)
        else:
            self.standard_frame.pack_forget()
            self.gov_frame.pack(fill="x", pady=8)

    def _browse_standard(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")])
        if path:
            self.standard_var.set(path)

    def _browse_crime(self) -> None:
        for path in filedialog.askopenfilenames(filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")]):
            if path not in self.crime_files:
                self.crime_files.append(path)
                self.crime_list.insert("end", path)

    def _browse_pop(self) -> None:
        for path in filedialog.askopenfilenames(filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")]):
            if path not in self.pop_files:
                self.pop_files.append(path)
                self.pop_list.insert("end", path)

    def _browse_save(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=DEFAULT_PROCESSED_PATH.name,
        )
        if path:
            self.path_var.set(path)

    def _params(self) -> UploadParams:
        if self.mode_var.get() == "standard":
            return UploadParams(mode="standard", standard_file=self.standard_var.get().strip())
        return UploadParams(mode="government", crime_files=list(self.crime_files), pop_files=list(self.pop_files))

    def _on_process(self) -> None:
        if self._busy:
            return
        params = self._params()
        errors = params.validation_errors()
        if errors:
            messagebox.showerror("입력 오류", "\n".join(errors))
            return
        self.set_busy(True)
        self.log("업로드 전처리 시작")
        threading.Thread(target=self._run_process, args=(params,), daemon=True).start()

    def _run_process(self, params: UploadParams) -> None:
        try:
            df = run_excel_pipeline(params)
            self._df = df
            self.root.after(0, self._show_dataframe, df)
            self.log(f"업로드 전처리 완료: {len(df):,}행")
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("전처리 실패", str(exc)))
            self.log(f"업로드 전처리 실패: {exc}")
        finally:
            self.root.after(0, lambda: self.set_busy(False))

    def _show_dataframe(self, df: pd.DataFrame) -> None:
        self.clear_table(self.tree)
        for _, row in df.head(10).iterrows():
            self.tree.insert(
                "",
                "end",
                values=[
                    row.get(COL_CRIME, ""),
                    row.get(COL_REGION, ""),
                    row.get(COL_YEAR, ""),
                    self.format_number(row.get(COL_INCIDENTS, "")),
                    self.format_number(row.get(COL_POP, "")),
                    self.format_number(row.get(COL_RATE, ""), 2),
                ],
            )

    def _on_save(self) -> None:
        if self._df is None:
            messagebox.showwarning("저장 불가", "먼저 데이터를 처리하세요.")
            return
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("경로 없음", "저장 경로를 입력하세요.")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if DataExporter.save_to_csv(self._df, path):
            self.log(f"전처리 CSV 저장 완료: {path}")
            messagebox.showinfo("저장 완료", path)
        else:
            messagebox.showerror("저장 실패", path)


class FilePredictionTab(BaseTab):
    def __init__(self, parent: ttk.Notebook, app: "CrimePredictionApp") -> None:
        super().__init__(parent, app)
        self.input_var = tk.StringVar()
        self._build()

    def _build(self) -> None:
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=12, pady=12)
        _label(top, "파일 예측", font=FONT_TITLE, bg=BG).pack(anchor="w")
        _label(
            top,
            "예측 결과는 data/prediction_result.xlsx에 저장됩니다. Excel 파일은 VS Code가 아니라 Excel 프로그램으로 여는 것을 권장합니다.",
            fg=TEXT_SEC,
            bg=BG,
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))
        row = tk.Frame(top, bg=BG)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.input_var, font=FONT_SMALL).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="파일 선택", command=self._browse_input, relief="flat", bg=BORDER).pack(side="left", padx=6)
        self.add_button(_btn(row, "예측 실행", self._on_predict, bg=ACCENT)).pack(side="left")
        self.add_button(_btn(row, "결과 열기", self._open_result, bg=SUCCESS)).pack(side="left", padx=(6, 0))

        body = _card(self)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        inner = tk.Frame(body, bg=PANEL_BG, padx=14, pady=10)
        inner.pack(fill="both", expand=True)
        self.tree = self.make_table(
            inner,
            [COL_YEAR, COL_REGION, COL_CRIME, COL_POP, PREDICTED_INCIDENTS_COLUMN, PREDICTED_RATE_COLUMN],
        )

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")])
        if path:
            self.input_var.set(path)

    def _on_predict(self) -> None:
        if self._busy:
            return
        path = self.input_var.get().strip()
        if not path:
            messagebox.showwarning("입력 없음", "예측할 CSV/XLSX 파일을 선택하세요.")
            return
        self.set_busy(True)
        self.log(f"파일 예측 시작: {path}")
        threading.Thread(target=self._run_predict, args=(path,), daemon=True).start()

    def _run_predict(self, path: str) -> None:
        try:
            df = predict_from_file(path, PREDICTION_OUTPUT_PATH)
            self.root.after(0, self._show_dataframe, df)
            self.log(f"파일 예측 완료: {PREDICTION_OUTPUT_PATH}")
            self.root.after(0, lambda: messagebox.showinfo("예측 완료", f"결과 저장:\n{PREDICTION_OUTPUT_PATH}"))
        except Exception as exc:
            message = friendly_prediction_error(str(exc))
            self.log(f"파일 예측 실패: {message}")
            self.root.after(0, lambda: messagebox.showerror("예측 실패", message))
        finally:
            self.root.after(0, lambda: self.set_busy(False))

    def _show_dataframe(self, df: pd.DataFrame) -> None:
        self.clear_table(self.tree)
        for _, row in df.head(10).iterrows():
            self.tree.insert(
                "",
                "end",
                values=[
                    row.get(COL_YEAR, ""),
                    row.get(COL_REGION, ""),
                    row.get(COL_CRIME, ""),
                    self.format_number(row.get(COL_POP, "")),
                    self.format_number(row.get(PREDICTED_INCIDENTS_COLUMN, ""), 2),
                    self.format_number(row.get(PREDICTED_RATE_COLUMN, ""), 2),
                ],
            )

    def _open_result(self) -> None:
        if not PREDICTION_OUTPUT_PATH.exists():
            messagebox.showwarning("결과 없음", "먼저 파일 예측을 실행하세요.")
            return
        try:
            subprocess.Popen(["explorer", f"/select,{PREDICTION_OUTPUT_PATH}"])
        except Exception as exc:
            messagebox.showerror("열기 실패", str(exc))


class SinglePredictionTab(BaseTab):
    def __init__(self, parent: ttk.Notebook, app: "CrimePredictionApp") -> None:
        super().__init__(parent, app)
        self._build()

    def _build(self) -> None:
        card = _card(self)
        card.pack(fill="x", padx=12, pady=12)
        inner = tk.Frame(card, bg=PANEL_BG, padx=14, pady=14)
        inner.pack(fill="x")
        _label(inner, "단일 예측", font=FONT_TITLE).pack(anchor="w", pady=(0, 10))

        self.year_var = tk.IntVar(value=2025)
        self.region_var = tk.StringVar(value="서울")
        self.crime_var = tk.StringVar(value="절도")
        self.population_var = tk.IntVar(value=9_000_000)
        self._input_row(inner, "연도", self.year_var)
        self._input_row(inner, "지역", self.region_var)
        self._input_row(inner, "범죄_유형", self.crime_var)
        self._input_row(inner, "인구수", self.population_var)
        self.add_button(_btn(inner, "예측", self._on_predict, bg=ACCENT)).pack(anchor="w", pady=(10, 0))

        result_card = _card(self)
        result_card.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.result_text = tk.Text(result_card, font=FONT_MONO, wrap="word", height=12)
        self.result_text.pack(fill="both", expand=True, padx=10, pady=10)

    def _input_row(self, parent: tk.Widget, label: str, variable) -> None:
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", pady=3)
        _label(row, label, fg=TEXT_SEC, width=10, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=variable, font=FONT_SMALL, width=22).pack(side="left")

    def _on_predict(self) -> None:
        if self._busy:
            return
        try:
            year = int(self.year_var.get())
            region = self.region_var.get().strip()
            crime_type = self.crime_var.get().strip()
            population = int(self.population_var.get())
            if not region or not crime_type:
                raise ValueError("지역과 범죄_유형을 입력하세요.")
            if population <= 0:
                raise ValueError("인구수는 1 이상이어야 합니다.")
        except Exception as exc:
            messagebox.showerror("입력 오류", str(exc))
            return

        self.set_busy(True)
        threading.Thread(
            target=self._run_predict,
            args=(year, region, crime_type, population),
            daemon=True,
        ).start()

    def _run_predict(self, year: int, region: str, crime_type: str, population: int) -> None:
        try:
            predicted_incidents = predict_one(year, region, crime_type, population)
            predicted_rate = predicted_incidents / population * 100000
            message = (
                f"{year}년 {region} {crime_type} 예측 결과: "
                f"발생 건수 {predicted_incidents:,.2f}건, 범죄율 {predicted_rate:.2f}"
            )
            self.root.after(0, self._append_result, message)
            self.log(message)
        except Exception as exc:
            message = friendly_prediction_error(str(exc))
            self.root.after(0, lambda: messagebox.showerror("단일 예측 실패", message))
            self.log(f"단일 예측 실패: {message}")
        finally:
            self.root.after(0, lambda: self.set_busy(False))

    def _append_result(self, message: str) -> None:
        self.result_text.insert("end", message + "\n")
        self.result_text.see("end")


class LogTab(BaseTab):
    def __init__(self, parent: ttk.Notebook, app: "CrimePredictionApp") -> None:
        super().__init__(parent, app)
        self.text = tk.Text(self, font=FONT_MONO, wrap="word")
        self.text.pack(fill="both", expand=True, padx=12, pady=12)


class CrimePredictionApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Crime Prediction 통합 GUI")
        self.root.geometry("1100x760")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        style = ttk.Style(self.root)
        style.configure("App.TFrame", background=BG)

        header = tk.Frame(self.root, bg=ACCENT, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Crime Prediction 통합 GUI",
            font=("맑은 고딕", 14, "bold"),
            fg="white",
            bg=ACCENT,
        ).pack(side="left", padx=18)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.log_tab = LogTab(self.notebook, self)
        self.logger = TextLogger(self.root, self.log_tab.text)
        self._add_tab(DataGenerationTab, "데이터 생성")
        self._add_tab(UploadPreprocessTab, "Excel/CSV 업로드 및 전처리")
        self._add_tab(FilePredictionTab, "파일 예측")
        self._add_tab(SinglePredictionTab, "단일 예측")
        self.notebook.add(self.log_tab, text="결과 확인/로그")
        self.log("통합 GUI 준비 완료")

    def _add_tab(self, cls, title: str) -> None:
        self.notebook.add(cls(self.notebook, self), text=title)

    def log(self, message: str) -> None:
        self.logger.write(message)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    CrimePredictionApp().run()
