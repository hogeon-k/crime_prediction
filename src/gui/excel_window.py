from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

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
C_CRIME = "\ubc94\uc8c4_\uc720\ud615"
C_REGION = "\uc9c0\uc5ed"
C_YEAR = "\uc5f0\ub3c4"
C_INC = "\ubc1c\uc0dd_\uac74\uc218"
C_POP = "\uc778\uad6c\uc218"
C_RATE = "\ubc94\uc8c4\uc728"


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

        self._build_ui()

    def _build_ui(self) -> None:
        hdr = tk.Frame(self.root, bg=ACCENT, pady=10)
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="\U0001f4ca  Excel / CSV \ud559\uc2b5 \ub370\uc774\ud130 \uc5c5\ub85c\ub4dc",
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
            value=os.path.join(os.getcwd(), "processed_crime_data.csv")
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
        _btn(
            btn_frame,
            "\u25b6  \uc5c5\ub85c\ub4dc & \ucc98\ub9ac",
            self._on_process,
            bg=ACCENT,
        ).pack(fill="x", pady=(0, 6))
        _btn(btn_frame, "\U0001f4be  CSV \uc800\uc7a5", self._on_save, bg=SUCCESS).pack(
            fill="x"
        )

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
        cols = [C_CRIME, C_REGION, C_YEAR, C_INC, C_POP, C_RATE]
        self._tree = ttk.Treeview(parent, columns=cols, show="headings", height=10)
        widths = {
            C_CRIME: 130,
            C_REGION: 70,
            C_YEAR: 55,
            C_INC: 80,
            C_POP: 90,
            C_RATE: 75,
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
        params = self._build_params()
        errors = params.validation_errors()
        if errors:
            messagebox.showerror("\uc785\ub825 \uc624\ub958", "\n".join(errors))
            return

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
        self._set_status("\u2705  \ucc98\ub9ac \uc644\ub8cc", SUCCESS)
        if self._df is not None:
            self._update_stats(self._df)
            self._populate_table(self._df)

    def _on_fail(self, step: str, message: str) -> None:
        self._progress.stop()
        self._set_status(f"\u274c  \uc2e4\ud328: {step}", DANGER)
        messagebox.showerror(
            "\ucc98\ub9ac \uc2e4\ud328", f"\ub2e8\uacc4: {step}\n\n{message}"
        )

    def _on_save(self) -> None:
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

    def _set_status(self, msg: str, color: str = TEXT_PRI) -> None:
        self._status_var.set(msg)
        self._status_lbl.config(fg=color)

    def _update_stats(self, df: pd.DataFrame) -> None:
        self._stat_labels["rows"].config(text=f"{len(df):,} \ud589")
        self._stat_labels["cols"].config(text=f"{len(df.columns)} \uc5f4")
        self._stat_labels["crime_types"].config(text=str(df[C_CRIME].nunique()))
        self._stat_labels["regions"].config(text=str(df[C_REGION].nunique()))
        years = sorted(df[C_YEAR].dropna().unique())
        self._stat_labels["year_range"].config(
            text=f"{int(years[0])} ~ {int(years[-1])}" if len(years) else "\u2014"
        )
        avg = df[C_RATE].mean() if C_RATE in df.columns else 0
        self._stat_labels["crime_rate"].config(text=f"{avg:.2f}")

    def _clear_table(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

    def _populate_table(self, df: pd.DataFrame) -> None:
        self._clear_table()
        preview = DataExporter.preview(df, rows=10)
        for i, row in enumerate(preview["data"]):
            tag = "even" if i % 2 == 0 else "odd"
            vals = [
                row.get(C_CRIME, ""),
                row.get(C_REGION, ""),
                row.get(C_YEAR, ""),
                f"{row.get(C_INC, 0):,}",
                f"{row.get(C_POP, 0):,}",
                f"{row.get(C_RATE, 0):.2f}",
            ]
            self._tree.insert("", "end", values=vals, tags=(tag,))
        self._tree.tag_configure("even", background="#F9FAFB")
        self._tree.tag_configure("odd", background=PANEL_BG)

    def run(self) -> None:
        self.root.mainloop()
