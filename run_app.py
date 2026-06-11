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

from constants import PREDICTED_INCIDENTS_COLUMN, PREDICTED_RATE_COLUMN  # noqa: E402
from constants import BASE_YEAR_COLUMN, TARGET_YEAR_COLUMN  # noqa: E402
from gui.widgets import (  # noqa: E402
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
    button as _btn,
    card as _card,
    label as _label,
)
from gui.excel_window import ExcelWindow  # noqa: E402
from model.excel_model import UploadParams  # noqa: E402
from services.analysis_data_service import ACTUAL_KIND, DATA_KIND_COLUMN, PREDICTED_KIND  # noqa: E402
from services.dummy_generator import (  # noqa: E402
    VALID_REGIONS,
    DataExporter,
    run_generation_pipeline,
)
from viewmodel.crime_viewmodel import CrimeViewModel  # noqa: E402

DEFAULT_GENERATED_PATH = SRC_DIR / "data" / "generated_crime_data.csv"
DEFAULT_PROCESSED_PATH = SRC_DIR / "data" / "processed_crime_data.csv"
DEFAULT_ANALYSIS_PATH = SRC_DIR / "data" / "processed_crime_data.csv"
ANALYSIS_LOAD_MODES = {
    "실제 + 예측 데이터 통합": "actual_predicted",
    "실제 데이터만": "actual_only",
    "예측 결과만": "predicted_only",
    "직접 파일 선택": "manual_file",
}

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
        self.viewmodel = CrimeViewModel(callback=lambda _state: None)
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
        result = DataExporter.save_to_csv(self._df, path)
        if result:
            self.log(f"더미 CSV 저장 완료: {path}")
            messagebox.showinfo("저장 완료", path)
        else:
            messagebox.showerror("저장 실패", result.message)


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
        df = self.viewmodel.process_upload(params)
        if df is not None:
            self._df = df
            self.root.after(0, self._show_dataframe, df)
            self.log(f"업로드 전처리 완료: {len(df):,}행")
        else:
            message = self.viewmodel.state.error_message
            self.root.after(0, lambda: messagebox.showerror("전처리 실패", message))
            self.log(f"업로드 전처리 실패: {message}")
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
        result = DataExporter.save_to_csv(self._df, path)
        if result:
            self.log(f"전처리 CSV 저장 완료: {path}")
            messagebox.showinfo("저장 완료", path)
        else:
            messagebox.showerror("저장 실패", result.message)


class FilePredictionTab(BaseTab):
    def __init__(self, parent: ttk.Notebook, app: "CrimePredictionApp") -> None:
        super().__init__(parent, app)
        self.input_var = tk.StringVar()
        self.target_year_var = tk.IntVar(value=2025)
        self.stat_labels: dict[str, tk.Label] = {}
        self._chart_canvases = []
        self._last_output_path: Path | None = None
        self._build()

    def _build(self) -> None:
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=12, pady=12)
        _label(top, "파일 예측", font=FONT_TITLE, bg=BG).pack(anchor="w")
        _label(
            top,
            "예측 결과는 data/prediction_result_YYYY.xlsx 형식으로 연도별 저장됩니다. Excel 파일은 Excel 프로그램으로 여는 것을 권장합니다.",
            fg=TEXT_SEC,
            bg=BG,
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))
        row = tk.Frame(top, bg=BG)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.input_var, font=FONT_SMALL).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="파일 선택", command=self._browse_input, relief="flat", bg=BORDER).pack(side="left", padx=6)
        _label(row, "예측 대상 연도", fg=TEXT_SEC, bg=BG).pack(side="left", padx=(8, 4))
        ttk.Entry(row, textvariable=self.target_year_var, font=FONT_SMALL, width=8).pack(side="left")
        self.add_button(_btn(row, "예측 실행", self._on_predict, bg=ACCENT)).pack(side="left")
        self.add_button(_btn(row, "결과 열기", self._open_result, bg=SUCCESS)).pack(side="left", padx=(6, 0))
        self.status_var = tk.StringVar(value="파일을 선택하세요.")
        _label(top, "", fg=TEXT_SEC, bg=BG, textvariable=self.status_var).pack(
            anchor="w", pady=(6, 0)
        )

        body = _card(self)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        inner = tk.Frame(body, bg=PANEL_BG, padx=14, pady=10)
        inner.pack(fill="both", expand=True)

        stats = tk.Frame(inner, bg=PANEL_BG)
        stats.pack(fill="x", pady=(0, 8))
        for key, title in [
            ("avg_predicted_incidents", "평균 예측 발생 건수"),
            ("max_predicted_incidents", "최대 예측 발생 건수"),
            ("avg_predicted_rate", "평균 예측 범죄율"),
            ("region_count", "지역 수"),
            ("crime_type_count", "범죄 유형 수"),
        ]:
            item = tk.Frame(stats, bg=PANEL_BG)
            item.pack(side="left", fill="x", expand=True, padx=(0, 8))
            _label(item, title, fg=TEXT_SEC).pack(anchor="w")
            value_label = _label(item, "\u2014", font=FONT_BOLD, fg=TEXT_PRI)
            value_label.pack(anchor="w")
            self.stat_labels[key] = value_label

        self.tree = self.make_table(
            inner,
            [
                BASE_YEAR_COLUMN,
                TARGET_YEAR_COLUMN,
                COL_REGION,
                COL_CRIME,
                COL_POP,
                PREDICTED_INCIDENTS_COLUMN,
                PREDICTED_RATE_COLUMN,
            ],
        )

        chart_frame = tk.Frame(inner, bg=PANEL_BG)
        chart_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.chart_notebook = ttk.Notebook(chart_frame)
        self.chart_notebook.pack(fill="both", expand=True)
        self.region_chart_frame = tk.Frame(self.chart_notebook, bg=PANEL_BG)
        self.crime_chart_frame = tk.Frame(self.chart_notebook, bg=PANEL_BG)
        self.rate_chart_frame = tk.Frame(self.chart_notebook, bg=PANEL_BG)
        self.chart_notebook.add(self.region_chart_frame, text="지역별")
        self.chart_notebook.add(self.crime_chart_frame, text="범죄 유형별")
        self.chart_notebook.add(self.rate_chart_frame, text="범죄율 TOP 10")

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")])
        if path:
            self.input_var.set(path)
            self.status_var.set("파일 선택 완료")
            self.log(f"파일 선택 완료: {path}")

    def _on_predict(self) -> None:
        if self._busy:
            return
        path = self.input_var.get().strip()
        if not path:
            messagebox.showwarning("입력 없음", "예측할 CSV/XLSX 파일을 선택하세요.")
            return
        try:
            target_year = int(self.target_year_var.get())
        except Exception:
            messagebox.showerror("입력 오류", "예측 대상 연도는 숫자여야 합니다.")
            return
        self.set_busy(True)
        self.status_var.set("데이터 검증 중")
        self.log(f"파일 예측 시작: {path}, 예측 대상 연도={target_year}")
        threading.Thread(target=self._run_predict, args=(path, target_year), daemon=True).start()

    def _run_predict(self, path: str, target_year: int) -> None:
        self.root.after(0, lambda: self.status_var.set("모델 로드 중"))
        output_path = self.viewmodel.prediction_result_path(target_year)
        df = self.viewmodel.predict_file(path, str(output_path), target_year=target_year)
        if df is not None:
            self._last_output_path = output_path
            self.root.after(0, lambda: self.status_var.set("예측 결과 표시 중"))
            self.root.after(0, self._show_dataframe, df)
            self.root.after(0, self._show_statistics)
            self.root.after(0, self._show_charts)
            self.log(f"파일 예측 완료: {output_path}")
            self.root.after(0, lambda: self.status_var.set("예측 완료 / 결과 저장 완료"))
            self.root.after(
                0,
                lambda: messagebox.showinfo("예측 완료", f"결과 저장 완료:\n{output_path}"),
            )
        else:
            message = self.viewmodel.state.error_message
            self.log(f"파일 예측 실패: {message}")
            self.root.after(0, lambda m=message: self.status_var.set(m.splitlines()[0]))
            self.root.after(0, lambda: messagebox.showerror("예측 실패", message))
        self.root.after(0, lambda: self.set_busy(False))

    def _show_dataframe(self, df: pd.DataFrame) -> None:
        self.clear_table(self.tree)
        for _, row in df.head(10).iterrows():
            self.tree.insert(
                "",
                "end",
                values=[
                    row.get(BASE_YEAR_COLUMN, ""),
                    row.get(TARGET_YEAR_COLUMN, ""),
                    row.get(COL_REGION, ""),
                    row.get(COL_CRIME, ""),
                    self.format_number(row.get(COL_POP, "")),
                    self.format_number(row.get(PREDICTED_INCIDENTS_COLUMN, ""), 2),
                    self.format_number(row.get(PREDICTED_RATE_COLUMN, ""), 2),
                ],
            )

    def _show_statistics(self) -> None:
        summary = self.viewmodel.get_prediction_summary()

        def fmt(value, digits: int = 2) -> str:
            if value is None:
                return "\u2014"
            return f"{float(value):,.{digits}f}"

        self.stat_labels["avg_predicted_incidents"].config(
            text=fmt(summary.get("avg_predicted_incidents"))
        )
        self.stat_labels["max_predicted_incidents"].config(
            text=fmt(summary.get("max_predicted_incidents"))
        )
        self.stat_labels["avg_predicted_rate"].config(
            text=fmt(summary.get("avg_predicted_rate"))
        )
        self.stat_labels["region_count"].config(text=str(summary.get("region_count", 0)))
        self.stat_labels["crime_type_count"].config(text=str(summary.get("crime_type_count", 0)))

    def _show_charts(self) -> None:
        self._draw_bar_chart(
            self.region_chart_frame,
            self.viewmodel.get_region_chart_data(),
            "지역별 예측 발생 건수",
        )
        self._draw_bar_chart(
            self.crime_chart_frame,
            self.viewmodel.get_crime_type_chart_data(),
            "범죄 유형별 예측 발생 건수",
        )
        self._draw_bar_chart(
            self.rate_chart_frame,
            self.viewmodel.get_top_rate_chart_data(),
            "예측 범죄율 TOP 10",
        )

    def _draw_bar_chart(
        self,
        parent: tk.Frame,
        rows: list[dict[str, float | str]],
        title: str,
    ) -> None:
        for widget in parent.winfo_children():
            widget.destroy()

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            from matplotlib import font_manager
        except Exception as exc:
            _label(parent, f"그래프를 표시하려면 matplotlib이 필요합니다: {exc}", fg=DANGER).pack(
                anchor="w"
            )
            return

        if not rows:
            _label(parent, "표시할 데이터가 없습니다.", fg=TEXT_SEC).pack(anchor="w")
            return

        limited_rows = rows[:10]
        labels = [str(row["label"]) for row in limited_rows]
        values = [float(row["value"]) for row in limited_rows]
        font_path = Path("C:/Windows/Fonts/malgun.ttf")
        font_properties = (
            font_manager.FontProperties(fname=str(font_path))
            if font_path.exists()
            else None
        )

        fig = Figure(figsize=(7.5, 2.8), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(labels, values, color=ACCENT)
        ax.set_title(title, fontproperties=font_properties)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_properties)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._chart_canvases.append(canvas)

    def _open_result(self) -> None:
        path = self._last_output_path or self.viewmodel.prediction_result_path(int(self.target_year_var.get()))
        if not path.exists():
            messagebox.showwarning("결과 없음", "먼저 파일 예측을 실행하세요.")
            return
        try:
            subprocess.Popen(["explorer", f"/select,{path}"])
        except Exception as exc:
            messagebox.showerror("열기 실패", str(exc))


class AnalysisTab(BaseTab):
    REGION_POINTS = {
        "서울": (330, 120),
        "인천": (270, 135),
        "경기": (330, 165),
        "강원": (470, 125),
        "충북": (390, 235),
        "충남": (295, 265),
        "세종": (345, 275),
        "대전": (355, 310),
        "경북": (500, 305),
        "대구": (485, 365),
        "전북": (330, 385),
        "광주": (300, 475),
        "전남": (320, 530),
        "경남": (450, 455),
        "부산": (530, 465),
        "울산": (555, 415),
        "제주": (295, 650),
    }

    def __init__(self, parent: ttk.Notebook, app: "CrimePredictionApp") -> None:
        super().__init__(parent, app)
        self.region_var = tk.StringVar()
        self.year_var = tk.StringVar()
        self.region_query_var = tk.StringVar()
        self.crime_query_var = tk.StringVar()
        self.load_mode_var = tk.StringVar(value="실제 + 예측 데이터 통합")
        self.sort_var = tk.StringVar(value=COL_RATE)
        self.status_var = tk.StringVar(value="분석 데이터를 불러오세요.")
        self._chart_canvases = []
        self._build()
        self._load_default()

    def _build(self) -> None:
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=12, pady=12)
        _label(top, "지역 통계/지도 분석", font=FONT_TITLE, bg=BG).pack(anchor="w")

        controls = tk.Frame(top, bg=BG)
        controls.pack(fill="x", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.load_mode_var,
            values=list(ANALYSIS_LOAD_MODES),
            width=24,
            state="readonly",
        ).pack(side="left")
        self.add_button(_btn(controls, "분석 데이터 불러오기", self._load_selected_mode, bg=ACCENT)).pack(
            side="left", padx=(6, 0)
        )
        _label(controls, "지역", fg=TEXT_SEC, bg=BG).pack(side="left", padx=(10, 4))
        self.region_combo = ttk.Combobox(controls, textvariable=self.region_var, width=12, state="readonly")
        self.region_combo.pack(side="left")
        _label(controls, "연도", fg=TEXT_SEC, bg=BG).pack(side="left", padx=(10, 4))
        self.year_combo = ttk.Combobox(controls, textvariable=self.year_var, width=8, state="readonly")
        self.year_combo.pack(side="left")
        self.region_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh())
        self.year_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh())

        _label(controls, "지역 검색", fg=TEXT_SEC, bg=BG).pack(side="left", padx=(10, 4))
        ttk.Entry(controls, textvariable=self.region_query_var, width=12).pack(side="left")
        _label(controls, "범죄유형 검색", fg=TEXT_SEC, bg=BG).pack(side="left", padx=(10, 4))
        ttk.Entry(controls, textvariable=self.crime_query_var, width=14).pack(side="left")
        self.add_button(_btn(controls, "검색/갱신", self._refresh, bg=SUCCESS)).pack(side="left", padx=(8, 0))
        _label(top, "", fg=TEXT_SEC, bg=BG, textvariable=self.status_var).pack(anchor="w", pady=(6, 0))

        body = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=6)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left = _card(body)
        right = _card(body)
        body.add(left, minsize=420)
        body.add(right, minsize=500)

        left_inner = tk.Frame(left, bg=PANEL_BG, padx=10, pady=10)
        left_inner.pack(fill="both", expand=True)
        _label(left_inner, "지역별 범죄율 지도", font=FONT_BOLD).pack(anchor="w")
        _label(
            left_inner,
            "색이 붉을수록 범죄율이 높고, 선택 지역은 노란색으로 강조됩니다.",
            fg=TEXT_SEC,
        ).pack(anchor="w", pady=(2, 6))
        self.map_canvas = tk.Canvas(left_inner, width=620, height=700, bg="#F3F6FA", highlightthickness=0)
        self.map_canvas.pack(fill="both", expand=True, pady=(8, 0))

        right_inner = tk.Frame(right, bg=PANEL_BG, padx=10, pady=10)
        right_inner.pack(fill="both", expand=True)
        _label(right_inner, "범죄 리스트", font=FONT_BOLD).pack(anchor="w")
        sort_row = tk.Frame(right_inner, bg=PANEL_BG)
        sort_row.pack(fill="x", pady=(4, 6))
        _label(sort_row, "정렬", fg=TEXT_SEC).pack(side="left")
        ttk.Combobox(
            sort_row,
            textvariable=self.sort_var,
            values=[COL_INCIDENTS, COL_RATE, PREDICTED_INCIDENTS_COLUMN, PREDICTED_RATE_COLUMN],
            width=18,
            state="readonly",
        ).pack(side="left", padx=(4, 8))
        self.add_button(_btn(sort_row, "정렬 적용", self._refresh, bg=ACCENT)).pack(side="left")

        table_frame = tk.Frame(right_inner, bg=PANEL_BG)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, show="headings", height=10)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=vsb.set)
        self.table.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.trend_frame = tk.Frame(right_inner, bg=PANEL_BG)
        self.trend_frame.pack(fill="both", expand=True, pady=(10, 0))

    def _load_default(self) -> None:
        self._load_selected_mode()

    def _load_selected_mode(self) -> None:
        mode = ANALYSIS_LOAD_MODES.get(self.load_mode_var.get(), "actual_predicted")
        if mode == "actual_only":
            df = self.viewmodel.load_actual_analysis_data()
        elif mode == "predicted_only":
            df = self.viewmodel.load_prediction_results_by_year([2025, 2026, 2027])
        elif mode == "manual_file":
            self._browse_data()
            return
        else:
            df = self.viewmodel.load_combined_actual_prediction_data()

        if df is None:
            message = self.viewmodel.state.error_message
            self.status_var.set(message)
            messagebox.showerror("분석 데이터 로드 실패", message)
            return
        self._after_data_loaded("선택한 분석 데이터")

    def _browse_data(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")],
            initialdir=str(ROOT_DIR),
        )
        if path:
            self._load_data(Path(path))

    def _load_data(self, path: Path) -> None:
        df = self.viewmodel.load_dataframe_for_analysis(path)
        if df is None:
            message = self.viewmodel.state.error_message
            self.status_var.set(message)
            messagebox.showerror("분석 데이터 로드 실패", message)
            return
        self._after_data_loaded(str(path))

    def _after_data_loaded(self, label: str) -> None:
        regions = ["전체"] + self.viewmodel.get_available_regions()
        years = ["전체"] + [str(year) for year in self.viewmodel.get_available_years()]
        self.region_combo.config(values=regions)
        self.year_combo.config(values=years)
        self.region_var.set(regions[0] if regions else "")
        self.year_var.set(years[0] if years else "")
        messages = " ".join(self.viewmodel.analysis_messages)
        suffix = f" {messages} 현재 사용 가능한 연도만 표시합니다." if messages else ""
        self.status_var.set(f"분석 데이터 로드 완료: {label}.{suffix}")
        self._refresh()

    def _refresh(self) -> None:
        region = self.region_var.get()
        year_text = self.year_var.get()
        selected_region = None if region in ("", "전체") else region
        selected_year = None if year_text in ("", "전체") else int(year_text)
        self.viewmodel.set_region_year_selection(selected_region, selected_year)

        filtered = self.viewmodel.get_filtered_records(
            region_query=self.region_query_var.get().strip(),
            crime_type_query=self.crime_query_var.get().strip(),
            sort_by=self.sort_var.get(),
        )
        self._draw_map()
        self._populate_table(filtered)
        self._draw_trend()
        messages = " ".join(self.viewmodel.analysis_messages)
        suffix = f" {messages} 현재 사용 가능한 연도만 표시합니다." if messages else ""
        self.status_var.set(f"갱신 완료: {len(filtered):,}행.{suffix}")

    def _populate_table(self, df: pd.DataFrame) -> None:
        columns = self.viewmodel.get_table_columns()
        self.table.delete(*self.table.get_children())
        self.table.config(columns=columns)
        for column in columns:
            self.table.heading(column, text=column)
            self.table.column(column, width=105, anchor="center")
        for _, row in df.head(200).iterrows():
            values = []
            for column in columns:
                value = row.get(column, "")
                if column in (COL_INCIDENTS, COL_POP, PREDICTED_INCIDENTS_COLUMN):
                    value = self.format_number(value)
                elif column in (COL_RATE, PREDICTED_RATE_COLUMN):
                    value = self.format_number(value, 2)
                values.append(value)
            self.table.insert("", "end", values=values)

    def _draw_map(self) -> None:
        self.map_canvas.delete("all")
        rows = self.viewmodel.get_region_rate_map_data()
        if not rows:
            self.map_canvas.create_text(20, 20, anchor="nw", text="표시할 지도 데이터가 없습니다.", fill=TEXT_SEC)
            return

        values = [float(row["crime_rate"]) for row in rows]
        low, high = min(values), max(values)
        by_region = {str(row["region"]): float(row["crime_rate"]) for row in rows}
        selected = self.viewmodel.selected_region
        self.map_canvas.create_rectangle(14, 14, 606, 686, fill="#FFFFFF", outline="#D8DEE8", width=1)
        self.map_canvas.create_text(
            30,
            32,
            anchor="nw",
            text="지역별 평균 범죄율",
            fill=TEXT_PRI,
            font=("맑은 고딕", 11, "bold"),
        )
        self._draw_map_legend(low, high)
        self.map_canvas.create_line(80, 86, 540, 86, fill="#E5E7EB")

        for region, (x, y) in self.REGION_POINTS.items():
            value = by_region.get(region)
            if value is None:
                color = "#E5E7EB"
                rate_text = "-"
            else:
                color = self._rate_color(value, low, high)
                rate_text = f"{value:.2f}"

            is_selected = selected == region
            if is_selected:
                color = "#FACC15"
            outline = "#111827" if is_selected else "#CBD5E1"
            width = 3 if is_selected else 1
            self.map_canvas.create_oval(
                x - 25,
                y - 25,
                x + 25,
                y + 25,
                fill=color,
                outline=outline,
                width=width,
            )
            self.map_canvas.create_text(
                x,
                y - 6,
                text=region,
                font=("맑은 고딕", 8, "bold"),
                fill="#111827",
            )
            self.map_canvas.create_text(
                x,
                y + 10,
                text=rate_text,
                font=("맑은 고딕", 8),
                fill="#111827",
            )

    def _draw_map_legend(self, low: float, high: float) -> None:
        x0, y0 = 365, 34
        steps = 6
        for index in range(steps):
            ratio_value = low + (high - low) * (index / max(steps - 1, 1))
            color = self._rate_color(ratio_value, low, high)
            self.map_canvas.create_rectangle(
                x0 + index * 22,
                y0,
                x0 + (index + 1) * 22,
                y0 + 12,
                fill=color,
                outline=color,
            )
        self.map_canvas.create_text(x0, y0 + 20, anchor="nw", text=f"낮음 {low:.2f}", fill=TEXT_SEC, font=("맑은 고딕", 8))
        self.map_canvas.create_text(x0 + 132, y0 + 20, anchor="ne", text=f"높음 {high:.2f}", fill=TEXT_SEC, font=("맑은 고딕", 8))
        self.map_canvas.create_oval(x0 + 158, y0 - 2, x0 + 174, y0 + 14, fill="#FACC15", outline="#111827")
        self.map_canvas.create_text(x0 + 180, y0 + 6, anchor="w", text="선택", fill=TEXT_SEC, font=("맑은 고딕", 8))

    @staticmethod
    def _rate_color(value: float, low: float, high: float) -> str:
        if high <= low:
            ratio = 0.5
        else:
            ratio = (value - low) / (high - low)
        red = int(59 + ratio * 180)
        blue = int(220 - ratio * 160)
        green = int(130 - ratio * 80)
        return f"#{red:02x}{green:02x}{blue:02x}"

    def _draw_trend(self) -> None:
        for widget in self.trend_frame.winfo_children():
            widget.destroy()
        summary = self.viewmodel.get_yearly_crime_rate_summary()
        if summary.empty:
            _label(self.trend_frame, "연도별 범죄율 그래프를 표시할 데이터가 없습니다.", fg=TEXT_SEC).pack(anchor="w")
            return
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            from matplotlib import font_manager
        except Exception as exc:
            _label(self.trend_frame, f"그래프 생성 실패: {exc}", fg=DANGER).pack(anchor="w")
            return

        font_path = Path("C:/Windows/Fonts/malgun.ttf")
        font_properties = (
            font_manager.FontProperties(fname=str(font_path))
            if font_path.exists()
            else None
        )
        fig = Figure(figsize=(5.8, 2.4), dpi=100)
        ax = fig.add_subplot(111)
        legend_names = {
            ACTUAL_KIND: "실제 데이터",
            PREDICTED_KIND: "예측 데이터",
        }
        colors = {
            ACTUAL_KIND: ACCENT,
            PREDICTED_KIND: SUCCESS,
        }
        for data_kind, group in summary.groupby(DATA_KIND_COLUMN):
            group = group.sort_values(COL_YEAR)
            ax.plot(
                group[COL_YEAR].tolist(),
                group[COL_RATE].tolist(),
                marker="o",
                linewidth=2,
                color=colors.get(str(data_kind), "#64748B"),
                label=legend_names.get(str(data_kind), str(data_kind)),
            )
        ax.set_xlim(2021.5, 2027.5)
        ax.set_xticks(list(range(2022, 2028)))
        ax.set_xlabel("연도", fontproperties=font_properties)
        ax.set_ylabel("평균 범죄율", fontproperties=font_properties)
        ax.set_title(
            "2022~2027 연도별 범죄율 비교",
            fontproperties=font_properties,
        )
        ax.legend(prop=font_properties)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_properties)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.trend_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._chart_canvases.append(canvas)


class ModelPerformanceTab(BaseTab):
    def __init__(self, parent: ttk.Notebook, app: "CrimePredictionApp") -> None:
        super().__init__(parent, app)
        self._chart_canvases = []
        self._build()
        self._refresh()

    def _build(self) -> None:
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=12, pady=12)
        _label(top, "모델 성능/리소스 및 로그", font=FONT_TITLE, bg=BG).pack(anchor="w")
        _label(
            top,
            "현재 화면에서는 최종 저장 모델인 Linear Regression 성능과 실행 로그를 함께 확인합니다.",
            fg=TEXT_SEC,
            bg=BG,
        ).pack(anchor="w", pady=(4, 8))
        self.add_button(_btn(top, "새로고침", self._refresh, bg=ACCENT)).pack(anchor="w")

        body = _card(self)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        inner = tk.Frame(body, bg=PANEL_BG, padx=14, pady=10)
        inner.pack(fill="both", expand=True)
        columns = ["model", "mse", "rmse", "mae", "r2", "inference_seconds", "cpu_usage_percent"]
        headings = {
            "model": "모델",
            "mse": "MSE",
            "rmse": "RMSE",
            "mae": "MAE",
            "r2": "R²",
            "inference_seconds": "추론 시간(초)",
            "cpu_usage_percent": "CPU 사용량(%)",
        }
        self.table = ttk.Treeview(inner, columns=columns, show="headings", height=5)
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=135, anchor="center")
        self.table.pack(fill="x")
        _label(inner, "결과 확인 / 로그", font=FONT_BOLD).pack(anchor="w", pady=(12, 4))
        self.text = tk.Text(inner, font=FONT_MONO, wrap="word", height=12)
        self.text.pack(fill="both", expand=True)

    def _refresh(self) -> None:
        rows = self.viewmodel.get_model_performance_rows()
        self.table.delete(*self.table.get_children())
        for row in rows:
            self.table.insert(
                "",
                "end",
                values=[
                    row.get("model"),
                    self._fmt(row.get("mse")),
                    self._fmt(row.get("rmse")),
                    self._fmt(row.get("mae")),
                    self._fmt(row.get("r2")),
                    self._fmt(row.get("inference_seconds")),
                    self._fmt(row.get("cpu_usage_percent")),
                ],
            )

    @staticmethod
    def _fmt(value) -> str:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.4f}"


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
        self.prev_incidents_var = tk.StringVar()
        self.prev_rate_var = tk.StringVar()
        self._input_row(inner, "연도", self.year_var)
        self._input_row(inner, "지역", self.region_var)
        self._input_row(inner, "범죄_유형", self.crime_var)
        self._input_row(inner, "인구수", self.population_var)
        self._input_row(inner, "전년도 발생건수", self.prev_incidents_var)
        self._input_row(inner, "전년도 범죄율", self.prev_rate_var)
        _label(
            inner,
            "전년도 값은 비워두면 학습 시 저장된 지역/범죄유형 통계로 자동 보완됩니다.",
            fg=TEXT_SEC,
        ).pack(anchor="w", pady=(4, 0))
        self.add_button(_btn(inner, "예측", self._on_predict, bg=ACCENT)).pack(anchor="w", pady=(10, 0))

        result_card = _card(self)
        result_card.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        _label(result_card, "모델 성능/리소스 정보", font=FONT_BOLD).pack(anchor="w", padx=10, pady=(10, 0))
        self.result_text = tk.Text(result_card, font=FONT_MONO, wrap="word", height=12)
        self.result_text.pack(fill="both", expand=True, padx=10, pady=10)
        self._render_performance_summary(status="예측 대기")

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
            if self.prev_incidents_var.get().strip():
                float(self.prev_incidents_var.get())
            if self.prev_rate_var.get().strip():
                float(self.prev_rate_var.get())
        except Exception as exc:
            messagebox.showerror("입력 오류", str(exc))
            return

        self.set_busy(True)
        self.log("단일 예측 실행 중")
        threading.Thread(
            target=self._run_predict,
            args=(year, region, crime_type, population),
            daemon=True,
        ).start()

    def _run_predict(self, year: int, region: str, crime_type: str, population: int) -> None:
        predicted_incidents = self.viewmodel.predict_one(year, region, crime_type, population)
        if predicted_incidents is not None:
            predicted_rate = self.viewmodel.state.predicted_rate or 0.0
            message = (
                f"{year}년 {region} {crime_type} 예측 결과: "
                f"발생 건수 {predicted_incidents:,.2f}건, 범죄율 {predicted_rate:.2f}"
            )
            self.root.after(0, lambda: self._render_performance_summary("예측 완료", message))
            self.log(message)
        else:
            message = self.viewmodel.state.error_message
            self.root.after(0, lambda: self._render_performance_summary("예측 실패", message))
            self.root.after(0, lambda: messagebox.showerror("단일 예측 실패", message))
            self.log(f"단일 예측 실패: {message}")
        self.root.after(0, lambda: self.set_busy(False))

    def _render_performance_summary(self, status: str, prediction_message: str = "") -> None:
        summary = self.viewmodel.get_model_performance_summary()

        def fmt(value, digits: int = 4) -> str:
            if value is None or pd.isna(value):
                return "-"
            return f"{float(value):.{digits}f}"

        model_name = summary.get("model") or "-"
        lines = [
            f"현재 모델: {model_name}",
            f"Test R²: {fmt(summary.get('r2'))}",
            f"Test RMSE: {fmt(summary.get('rmse'), 2)}",
            f"Test MAE: {fmt(summary.get('mae'), 2)}",
            f"Test MSE: {fmt(summary.get('mse'), 2)}",
            f"추론 시간: {fmt(summary.get('inference_seconds'), 3)}초",
            f"CPU 사용량: {fmt(summary.get('cpu_usage_percent'), 1)}%",
            f"상태: {status}",
        ]
        if prediction_message:
            lines.append("")
            lines.append(prediction_message)
        if summary.get("message"):
            lines.append("")
            lines.append(str(summary["message"]))

        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "\n".join(lines))
        self.result_text.see("end")


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

        self.performance_log_tab = ModelPerformanceTab(self.notebook, self)
        self.logger = TextLogger(self.root, self.performance_log_tab.text)
        self._add_tab(DataGenerationTab, "데이터 생성")
        self._add_tab(UploadPreprocessTab, "Excel/CSV 업로드 및 전처리")
        self._add_tab(FilePredictionTab, "파일 예측")
        self._add_tab(SinglePredictionTab, "단일 예측")
        self._add_tab(AnalysisTab, "지역 통계/지도 분석")
        self.notebook.add(self.performance_log_tab, text="모델 성능/리소스 · 로그")
        self.log("통합 GUI 준비 완료")

    def _add_tab(self, cls, title: str) -> None:
        self.notebook.add(cls(self.notebook, self), text=title)

    def log(self, message: str) -> None:
        self.logger.write(message)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    CrimePredictionApp().run()
