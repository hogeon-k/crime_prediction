from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from ai.predict import (
    PREDICTED_INCIDENTS_COLUMN,
    PREDICTED_RATE_COLUMN,
    predict_from_file,
)
from gui.crime_generator_window import (
    ACCENT,
    BG,
    BORDER,
    DANGER,
    FONT_BOLD,
    FONT_LABEL,
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
from model.excel_model import CrimeState, ProcessStatus, UploadParams
from services.dummy_generator import DataExporter
from services.excel_pipeline import run_excel_pipeline

STEP_MERGE = "\ub370\uc774\ud130 \ubcd1\ud569"
COL_CRIME = "\ubc94\uc8c4_\uc720\ud615"
COL_REGION = "\uc9c0\uc5ed"
COL_YEAR = "\uc5f0\ub3c4"
COL_INCIDENTS = "\ubc1c\uc0dd_\uac74\uc218"
COL_POP = "\uc778\uad6c\uc218"
COL_RATE = "\ubc94\uc8c4\uc728"
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DATA_DIR = ROOT_DIR / "src" / "data"
PREDICTION_OUTPUT_PATH = ROOT_DIR / "data" / "prediction_result.xlsx"
SAMPLE_PREDICTION_INPUT_PATH = ROOT_DIR / "data" / "sample_prediction_input.xlsx"


class ExcelWindow:
    """범죄 데이터 Excel/CSV 업로드 및 처리 UI를 담당하는 윈도우 클래스"""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("\ubc94\uc8c4 \ub370\uc774\ud130 Excel \uc5c5\ub85c\ub4dc")
        self.root.geometry("960x720")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._df: pd.DataFrame | None = None
        self._crime_files: list[str] = []
        self._pop_files: list[str] = []
        self._busy = False
        self._action_buttons: list[tk.Button] = []

        self._build_ui()

    def _build_ui(self) -> None:
        hdr = tk.Frame(self.root, bg=ACCENT, pady=10)
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="📊  Excel / CSV 데이터 처리 및 저장 모델 예측",
            font=("\ub9d1\uc740 \uace0\ub515", 13, "bold"),
            fg="white",
            bg=ACCENT,
        ).pack(side="left", padx=20)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        left = tk.Frame(body, bg=BG, width=360)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_upload_panel(left)
        self._build_result_panel(right)

    def _build_upload_panel(self, parent: tk.Frame) -> None:
        card = _card(parent)
        card.pack(fill="both", expand=True)

        inner = tk.Frame(card, bg=PANEL_BG, padx=16, pady=14)
        inner.pack(fill="both", expand=True)

        _label(inner, "\ud30c\uc77c \uc5c5\ub85c\ub4dc", font=FONT_TITLE).pack(
            anchor="w", pady=(0, 12)
        )

        self._mode_var = tk.StringVar(value="standard")
        mode_frame = tk.Frame(inner, bg=PANEL_BG)
        mode_frame.pack(fill="x", pady=(0, 10))

        ttk.Radiobutton(
            mode_frame,
            text="\ud45c\uc900 \uc591\uc2dd (\ub2e8\uc77c Excel/CSV)",
            variable=self._mode_var,
            value="standard",
            command=self._on_mode_change,
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_frame,
            text="\uacf5\uacf5\ub370\uc774\ud130 (\ubc94\uc8c4 + \uc778\uad6c)",
            variable=self._mode_var,
            value="government",
            command=self._on_mode_change,
        ).pack(anchor="w", pady=(4, 0))

        self._standard_frame = tk.Frame(inner, bg=PANEL_BG)
        self._standard_frame.pack(fill="x", pady=8)
        _label(
            self._standard_frame, "\ud45c\uc900 \uc591\uc2dd \ud30c\uc77c", fg=TEXT_SEC
        ).pack(anchor="w")
        row = tk.Frame(self._standard_frame, bg=PANEL_BG)
        row.pack(fill="x", pady=4)
        self._standard_var = tk.StringVar()
        ttk.Entry(row, textvariable=self._standard_var, font=FONT_SMALL).pack(
            side="left", fill="x", expand=True
        )
        tk.Button(
            row,
            text="\u2026",
            relief="flat",
            bg=BORDER,
            cursor="hand2",
            command=self._browse_standard,
            padx=6,
        ).pack(side="right")

        self._gov_frame = tk.Frame(inner, bg=PANEL_BG)
        self._build_gov_file_row(
            self._gov_frame,
            "\ubc94\uc8c4 \ud30c\uc77c",
            self._browse_crime,
            "_crime_list",
        )
        self._build_gov_file_row(
            self._gov_frame, "\uc778\uad6c \ud30c\uc77c", self._browse_pop, "_pop_list"
        )

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=14)

        _label(inner, "\uc800\uc7a5 \uacbd\ub85c", fg=TEXT_SEC).pack(
            anchor="w", pady=(0, 4)
        )
        path_row = tk.Frame(inner, bg=PANEL_BG)
        path_row.pack(fill="x")
        self._path_var = tk.StringVar(
            value=str(SRC_DATA_DIR / "processed_crime_data.csv")
        )
        ttk.Entry(path_row, textvariable=self._path_var, font=FONT_SMALL).pack(
            side="left", fill="x", expand=True
        )
        tk.Button(
            path_row,
            text="\u2026",
            relief="flat",
            bg=BORDER,
            cursor="hand2",
            command=self._browse_save_path,
            padx=6,
        ).pack(side="right")

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=14)

        btn_frame = tk.Frame(inner, bg=PANEL_BG)
        btn_frame.pack(fill="x")
        self._process_btn = _btn(
            btn_frame,
            "\u25b6  \uc5c5\ub85c\ub4dc & \ucc98\ub9ac",
            self._on_process,
            bg=ACCENT,
        )
        self._process_btn.pack(fill="x", pady=(0, 6))
        self._save_btn = _btn(
            btn_frame,
            "\U0001f4be  CSV \uc800\uc7a5",
            self._on_save,
            bg=SUCCESS,
        )
        self._save_btn.pack(fill="x")
        self._action_buttons.extend([self._process_btn, self._save_btn])

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=14)
        _label(inner, "저장된 AI 모델로 예측", font=FONT_BOLD).pack(
            anchor="w", pady=(0, 4)
        )
        _label(
            inner,
            "CSV/XLSX 파일의 연도, 지역, 범죄_유형, 인구수 컬럼으로 "
            "best_model.pkl 추론만 실행합니다.",
            fg=TEXT_SEC,
            wraplength=310,
            justify="left",
        ).pack(anchor="w", pady=(0, 6))
        self._predict_btn = _btn(
            inner,
            "🤖  저장된 모델로 예측 실행",
            self._on_predict_file,
            bg=ACCENT,
        )
        self._predict_btn.pack(fill="x", pady=(0, 6))
        self._sample_btn = _btn(
            inner,
            "예측 샘플 파일 생성",
            self._on_create_prediction_sample,
            bg=SUCCESS,
        )
        self._sample_btn.pack(fill="x", pady=(0, 6))
        self._action_buttons.extend([self._predict_btn, self._sample_btn])
        _label(
            inner,
            f"\uacb0\uacfc: {PREDICTION_OUTPUT_PATH}",
            fg=TEXT_SEC,
            wraplength=310,
            justify="left",
        ).pack(anchor="w")

        self._on_mode_change()

    def _build_gov_file_row(
        self, parent: tk.Frame, label: str, browse_cmd, list_attr: str
    ) -> None:
        _label(parent, label, fg=TEXT_SEC).pack(anchor="w", pady=(8, 4))
        lb_frame = tk.Frame(parent, bg=PANEL_BG)
        lb_frame.pack(fill="x")

        sb = ttk.Scrollbar(lb_frame, orient="vertical")
        lb = tk.Listbox(
            lb_frame,
            font=FONT_SMALL,
            height=4,
            yscrollcommand=sb.set,
            exportselection=False,
        )
        sb.config(command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.pack(side="left", fill="both", expand=True)
        setattr(self, list_attr, lb)

        btn_row = tk.Frame(parent, bg=PANEL_BG)
        btn_row.pack(fill="x", pady=(4, 0))
        tk.Button(
            btn_row,
            text="\ucd94\uac00",
            command=browse_cmd,
            font=FONT_SMALL,
            relief="flat",
            bg=BORDER,
            cursor="hand2",
            padx=6,
        ).pack(side="left", padx=(0, 4))
        tk.Button(
            btn_row,
            text="\uc81c\uac70",
            command=lambda: self._remove_selected(list_attr),
            font=FONT_SMALL,
            relief="flat",
            bg=BORDER,
            cursor="hand2",
            padx=6,
        ).pack(side="left")

    def _build_result_panel(self, parent: tk.Frame) -> None:
        status_card = _card(parent)
        status_card.pack(fill="x", pady=(0, 8))
        s_inner = tk.Frame(status_card, bg=PANEL_BG, padx=14, pady=10)
        s_inner.pack(fill="x")

        _label(s_inner, "\ucc98\ub9ac \uc0c1\ud0dc", font=FONT_BOLD).pack(anchor="w")
        self._status_var = tk.StringVar(value="\ub300\uae30 \uc911")
        self._status_lbl = _label(s_inner, "", fg=TEXT_SEC)
        self._status_lbl.config(textvariable=self._status_var)
        self._status_lbl.pack(anchor="w")
        self._progress = ttk.Progressbar(s_inner, mode="indeterminate", length=300)
        self._progress.pack(fill="x", pady=(6, 0))

        stat_card = _card(parent)
        stat_card.pack(fill="x", pady=(0, 8))
        st_inner = tk.Frame(stat_card, bg=PANEL_BG, padx=14, pady=10)
        st_inner.pack(fill="x")
        _label(st_inner, "\ub370\uc774\ud130 \ud1b5\uacc4", font=FONT_BOLD).pack(
            anchor="w", pady=(0, 6)
        )

        self._stat_labels: dict[str, tk.Label] = {}
        stat_frame = tk.Frame(st_inner, bg=PANEL_BG)
        stat_frame.pack(fill="x")
        for key, title in [
            ("rows", "\ucd1d \ud589\uc218"),
            ("cols", "\uc5f4 \uc218"),
            ("crime_types", "\ubc94\uc8c4 \uc720\ud615"),
            ("regions", "\uc9c0\uc5ed \uc218"),
            ("year_range", "\uc5f0\ub3c4 \ubc94\uc704"),
            ("crime_rate", "\ud3c9\uade0 \ubc94\uc8c4\uc728"),
        ]:
            row = tk.Frame(stat_frame, bg=PANEL_BG)
            row.pack(fill="x", pady=1)
            _label(row, title, fg=TEXT_SEC, width=12, anchor="w").pack(side="left")
            lbl = _label(row, "\u2014", fg=TEXT_PRI)
            lbl.pack(side="left")
            self._stat_labels[key] = lbl

        preview_card = _card(parent)
        preview_card.pack(fill="both", expand=True)
        p_inner = tk.Frame(preview_card, bg=PANEL_BG, padx=14, pady=10)
        p_inner.pack(fill="both", expand=True)
        _label(
            p_inner, "\ubbf8\ub9ac\ubcf4\uae30 (\uc0c1\uc704 10\ud589)", font=FONT_BOLD
        ).pack(anchor="w", pady=(0, 6))
        self._build_table(p_inner)

    def _build_table(self, parent: tk.Frame) -> None:
        cols = [
            COL_CRIME,
            COL_REGION,
            COL_YEAR,
            COL_INCIDENTS,
            COL_POP,
            COL_RATE,
            PREDICTED_INCIDENTS_COLUMN,
            PREDICTED_RATE_COLUMN,
        ]
        self._tree = ttk.Treeview(parent, columns=cols, show="headings", height=10)
        widths = {
            COL_CRIME: 130,
            COL_REGION: 70,
            COL_YEAR: 55,
            COL_INCIDENTS: 80,
            COL_POP: 90,
            COL_RATE: 75,
            PREDICTED_INCIDENTS_COLUMN: 105,
            PREDICTED_RATE_COLUMN: 95,
        }
        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=widths.get(c, 80), anchor="center")

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    def _on_mode_change(self) -> None:
        if self._mode_var.get() == "standard":
            self._standard_frame.pack(fill="x", pady=8)
            self._gov_frame.pack_forget()
        else:
            self._gov_frame.pack(fill="x", pady=8)
            self._standard_frame.pack_forget()

    def _browse_standard(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("Excel/CSV", "*.xlsx *.xls *.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("CSV", "*.csv"),
            ]
        )
        if path:
            self._standard_var.set(path)

    def _browse_crime(self) -> None:
        paths = filedialog.askopenfilenames(
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")]
        )
        for p in paths:
            if p not in self._crime_files:
                self._crime_files.append(p)
                self._crime_list.insert("end", p)

    def _browse_pop(self) -> None:
        paths = filedialog.askopenfilenames(
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")]
        )
        for p in paths:
            if p not in self._pop_files:
                self._pop_files.append(p)
                self._pop_list.insert("end", p)

    def _remove_selected(self, list_attr: str) -> None:
        lb: tk.Listbox = getattr(self, list_attr)
        selected = list(lb.curselection())
        if not selected:
            return
        files = self._crime_files if list_attr == "_crime_list" else self._pop_files
        for idx in reversed(selected):
            files.pop(idx)
            lb.delete(idx)

    def _browse_save_path(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialdir=str(SRC_DATA_DIR),
            initialfile="processed_crime_data.csv",
        )
        if path:
            self._path_var.set(path)

    def _build_params(self) -> UploadParams:
        mode = self._mode_var.get()
        if mode == "standard":
            return UploadParams(
                mode="standard",
                standard_file=self._standard_var.get().strip(),
            )
        return UploadParams(
            mode="government",
            crime_files=list(self._crime_files),
            pop_files=list(self._pop_files),
        )

    def _on_process(self) -> None:
        if self._busy:
            return

        params = self._build_params()
        errors = params.validation_errors()
        if errors:
            messagebox.showerror("\uc785\ub825 \uc624\ub958", "\n".join(errors))
            return

        self._set_busy(True)
        self._set_status("\ucc98\ub9ac \uc911\u2026", TEXT_SEC)
        self._progress.start(12)
        self._clear_table()
        threading.Thread(target=self._run_pipeline, args=(params,), daemon=True).start()

    def _run_pipeline(self, params: UploadParams) -> None:
        try:
            df = run_excel_pipeline(params, on_state_update=self._on_state_update)
            self._df = df
            self.root.after(0, self._on_success)
        except Exception as exc:
            msg = str(exc)

            self.root.after(0, lambda: self._on_fail("업로드", msg))

    def _on_state_update(self, state: CrimeState) -> None:
        if state.status == ProcessStatus.RUNNING:
            total = 4 if state.current_step == STEP_MERGE else 3
            msg = f"[{len(state.completed_steps)}/{total}] {state.current_step} \ucc98\ub9ac \uc911\u2026"
            self.root.after(0, lambda m=msg: self._set_status(m, TEXT_SEC))

    def _on_success(self) -> None:
        self._progress.stop()
        self._set_busy(False)
        self._set_status("\u2705  \ucc98\ub9ac \uc644\ub8cc", SUCCESS)
        if self._df is not None:
            self._update_stats(self._df)
            self._populate_table(self._df)

    def _on_fail(self, step: str, message: str) -> None:
        self._progress.stop()
        self._set_busy(False)
        self._set_status(f"\u274c  \uc2e4\ud328: {step}", DANGER)
        messagebox.showerror(
            "\ucc98\ub9ac \uc2e4\ud328", f"\ub2e8\uacc4: {step}\n\n{message}"
        )

    def _on_save(self) -> None:
        if self._busy:
            messagebox.showwarning(
                "처리 중",
                "현재 작업이 끝난 뒤 저장하세요.",
            )
            return

        if self._df is None:
            messagebox.showwarning(
                "\uc800\uc7a5 \ubd88\uac00",
                "\uba3c\uc800 \ub370\uc774\ud130\ub97c \ucc98\ub9ac\ud558\uc138\uc694.",
            )
            return
        path = self._path_var.get().strip()
        if not path:
            messagebox.showwarning(
                "\uacbd\ub85c \uc5c6\uc74c",
                "\uc800\uc7a5 \uacbd\ub85c\ub97c \uc785\ub825\ud558\uc138\uc694.",
            )
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ok = DataExporter.save_to_csv(self._df, path)
        if ok:
            messagebox.showinfo(
                "\uc800\uc7a5 \uc644\ub8cc", f"\uc800\uc7a5 \uc644\ub8cc:\n{path}"
            )
        else:
            messagebox.showerror(
                "\uc800\uc7a5 \uc2e4\ud328",
                f"\ud30c\uc77c \uc800\uc7a5\uc5d0 \uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4:\n{path}",
            )

    def _on_predict_file(self) -> None:
        if self._busy:
            return

        path = filedialog.askopenfilename(
            filetypes=[
                ("Excel/CSV", "*.xlsx *.xls *.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("CSV", "*.csv"),
            ]
        )
        if not path:
            return

        self._set_busy(True)
        self._set_status("저장된 모델로 예측 중…", TEXT_SEC)
        self._progress.start(12)
        self._clear_table()
        threading.Thread(
            target=self._run_prediction,
            args=(path,),
            daemon=True,
        ).start()

    def _on_create_prediction_sample(self) -> None:
        if self._busy:
            messagebox.showwarning(
                "처리 중",
                "현재 작업이 끝난 뒤 샘플 파일을 생성하세요.",
            )
            return

        try:
            SAMPLE_PREDICTION_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {
                    "연도": [2025, 2025, 2025, 2025, 2025],
                    "지역": ["서울", "부산", "대구", "인천", "광주"],
                    "범죄_유형": ["절도", "폭력", "강도", "사기", "강도"],
                    "인구수": [9000000, 3300000, 2400000, 2950000, 1400000],
                }
            )
            df.to_excel(
                SAMPLE_PREDICTION_INPUT_PATH,
                index=False,
                engine="openpyxl",
            )
            messagebox.showinfo(
                "예측 샘플 파일 생성 완료",
                f"샘플 파일을 저장했습니다:\n{SAMPLE_PREDICTION_INPUT_PATH}",
            )
        except Exception as exc:
            messagebox.showerror(
                "예측 샘플 파일 생성 실패",
                f"샘플 파일을 만들 수 없습니다:\n{exc}",
            )

    def _run_prediction(self, input_path: str) -> None:
        try:
            df = predict_from_file(input_path, PREDICTION_OUTPUT_PATH)
            self._df = df
            self.root.after(0, self._on_prediction_success)
        except FileNotFoundError as exc:
            message = self._format_prediction_error(str(exc))
            self.root.after(
                0, lambda m=message: self._on_fail("저장된 모델 예측", m)
            )
        except ValueError as exc:
            message = self._format_prediction_error(str(exc))
            self.root.after(
                0, lambda m=message: self._on_fail("저장된 모델 예측", m)
            )
        except Exception as exc:
            self.root.after(
                0, lambda m=str(exc): self._on_fail("저장된 모델 예측", m)
            )

    def _on_prediction_success(self) -> None:
        self._progress.stop()
        self._set_busy(False)
        self._set_status("✅  저장된 모델 예측 완료", SUCCESS)
        if self._df is not None:
            self._update_stats(self._df)
            self._populate_table(self._df)
        messagebox.showinfo(
            "저장된 모델 예측 완료",
            f"\uacb0\uacfc \ud30c\uc77c\uc744 \uc800\uc7a5\ud588\uc2b5\ub2c8\ub2e4:\n{PREDICTION_OUTPUT_PATH}",
        )

    def _set_status(self, msg: str, color: str = TEXT_PRI) -> None:
        self._status_var.set(msg)
        self._status_lbl.config(fg=color)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for button in self._action_buttons:
            button.config(state=state)

    @staticmethod
    def _format_prediction_error(message: str) -> str:
        if "best_model.pkl" in message:
            return "먼저 src/ai/train.py를 실행해 모델을 생성하세요."

        if "예측에 필요한 컬럼" in message:
            return (
                f"{message}\n\n"
                "필수 컬럼: 연도, 지역, 범죄_유형, 인구수\n"
                "예측 샘플 파일 생성 버튼으로 입력 양식을 확인할 수 있습니다."
            )

        if "학습 데이터에 없는 예측 입력값" in message:
            return (
                f"{message}\n\n"
                "입력 파일의 지역/범죄_유형 값을 학습 데이터와 같은 이름으로 맞춰주세요.\n"
                "예: 서울특별시는 서울로, 절도는 절도범죄로 자동 보정되지만 "
                "학습에 전혀 없는 값은 예측할 수 없습니다."
            )

        return message

    def _update_stats(self, df: pd.DataFrame) -> None:
        self._stat_labels["rows"].config(text=f"{len(df):,} \ud589")
        self._stat_labels["cols"].config(text=f"{len(df.columns)} \uc5f4")
        self._stat_labels["crime_types"].config(text=str(df[COL_CRIME].nunique()))
        self._stat_labels["regions"].config(text=str(df[COL_REGION].nunique()))
        years = sorted(df[COL_YEAR].dropna().unique())
        self._stat_labels["year_range"].config(
            text=f"{int(years[0])} ~ {int(years[-1])}" if len(years) else "\u2014"
        )
        avg = df[COL_RATE].mean() if COL_RATE in df.columns else 0
        self._stat_labels["crime_rate"].config(text=f"{avg:.2f}")

    def _clear_table(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

    def _populate_table(self, df: pd.DataFrame) -> None:
        self._clear_table()
        preview = DataExporter.preview(df, rows=10)

        def _format_number(value, digits: int | None = None) -> str:
            if value == "" or pd.isna(value):
                return ""
            if digits is None:
                return f"{value:,.0f}"
            return f"{value:,.{digits}f}"

        for i, row in enumerate(preview["data"]):
            tag = "even" if i % 2 == 0 else "odd"
            vals = [
                row.get(COL_CRIME, ""),
                row.get(COL_REGION, ""),
                row.get(COL_YEAR, ""),
                _format_number(row.get(COL_INCIDENTS, "")),
                _format_number(row.get(COL_POP, "")),
                _format_number(row.get(COL_RATE, ""), digits=2),
                _format_number(row.get(PREDICTED_INCIDENTS_COLUMN, ""), digits=2),
                _format_number(row.get(PREDICTED_RATE_COLUMN, ""), digits=2),
            ]
            self._tree.insert("", "end", values=vals, tags=(tag,))
        self._tree.tag_configure("even", background="#F9FAFB")
        self._tree.tag_configure("odd", background=PANEL_BG)

    def run(self) -> None:
        self.root.mainloop()
