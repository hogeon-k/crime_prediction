from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List

import pandas as pd

from model.excel_model import CrimeState, ProcessStatus
from services.dummy_generator import (
    VALID_REGIONS,
    DataExporter,
    DataGenerator,
    GenerationParams,
    get_user_input_gui,
)
from viewmodel.crime_viewmodel import CrimeViewModel

# ══════════════════════════════════════════════════════════════
#  색상 / 폰트 상수
# ══════════════════════════════════════════════════════════════
BG = "#F8F9FA"
PANEL_BG = "#FFFFFF"
ACCENT = "#4F46E5"
ACCENT_H = "#4338CA"
SUCCESS = "#059669"
DANGER = "#DC2626"
TEXT_PRI = "#111827"
TEXT_SEC = "#6B7280"
BORDER = "#E5E7EB"

FONT_TITLE = ("맑은 고딕", 14, "bold")
FONT_LABEL = ("맑은 고딕", 10)
FONT_BOLD = ("맑은 고딕", 10, "bold")
FONT_SMALL = ("맑은 고딕", 9)
FONT_MONO = ("Consolas", 9)


# ══════════════════════════════════════════════════════════════
#  유틸
# ══════════════════════════════════════════════════════════════
def _card(parent: tk.Widget, **kw) -> tk.Frame:
    """흰 배경 카드 프레임"""
    f = tk.Frame(
        parent,
        bg=PANEL_BG,
        relief="flat",
        highlightbackground=BORDER,
        highlightthickness=1,
        **kw,
    )
    return f


def _label(parent, text, font=None, fg=TEXT_PRI, bg=PANEL_BG, **kw):
    return tk.Label(parent, text=text, font=font or FONT_LABEL, fg=fg, bg=bg, **kw)


def _btn(parent, text, command, bg=ACCENT, fg="white", font=None, **kw):
    b = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        font=font or FONT_BOLD,
        relief="flat",
        cursor="hand2",
        activebackground=ACCENT_H,
        activeforeground="white",
        padx=12,
        pady=6,
        **kw,
    )
    b.bind("<Enter>", lambda e: b.config(bg=ACCENT_H))
    b.bind("<Leave>", lambda e: b.config(bg=bg))
    return b


# ══════════════════════════════════════════════════════════════
#  메인 윈도우
# ══════════════════════════════════════════════════════════════
class CrimeGeneratorWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("더미 범죄 데이터 생성")
        self.root.geometry("960x720")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._df: pd.DataFrame | None = None
        self._save_path: str = ""

        self._build_ui()

    # ──────────────────────────────────────────
    #  UI 조립
    # ──────────────────────────────────────────
    def _build_ui(self) -> None:
        # 타이틀 바
        hdr = tk.Frame(self.root, bg=ACCENT, pady=10)
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="🗂  더미 범죄 데이터 생성기",
            font=("맑은 고딕", 13, "bold"),
            fg="white",
            bg=ACCENT,
        ).pack(side="left", padx=20)

        # 본문 (좌: 설정, 우: 결과)
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        left = tk.Frame(body, bg=BG, width=340)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_param_panel(left)
        self._build_result_panel(right)

    # ── 왼쪽: 파라미터 패널 ────────────────────
    def _build_param_panel(self, parent: tk.Frame) -> None:
        card = _card(parent)
        card.pack(fill="both", expand=True)

        inner = tk.Frame(card, bg=PANEL_BG, padx=16, pady=14)
        inner.pack(fill="both", expand=True)

        _label(inner, "더미 데이터 생성 설정", font=FONT_TITLE).pack(
            anchor="w", pady=(0, 6)
        )
        _label(
            inner,
            "테스트와 시연에 사용할 샘플 범죄 데이터를 만듭니다. "
            "저장된 AI 모델 예측은 Excel/CSV 업로드 창에서 실행하세요.",
            fg=TEXT_SEC,
            wraplength=300,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        # 데이터 수
        self._build_spinbox_row(inner, "데이터 수", "data_count", 1, 10000, 500)

        # 시작 연도
        self._build_spinbox_row(inner, "시작 연도", "year_start", 2020, 2026, 2022)

        # 종료 연도
        self._build_spinbox_row(inner, "종료 연도", "year_end", 2020, 2026, 2024)

        # 지역 선택
        self._build_region_selector(inner)

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=14)

        # 저장 경로
        self._build_save_path_row(inner)

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=14)

        # 버튼
        btn_frame = tk.Frame(inner, bg=PANEL_BG)
        btn_frame.pack(fill="x")

        _btn(
            btn_frame,
            "▶  더미 데이터 생성 & 전처리",
            self._on_generate,
            bg=ACCENT,
        ).pack(fill="x", pady=(0, 6))

        _btn(btn_frame, "💾  더미 CSV 저장", self._on_save, bg=SUCCESS).pack(
            fill="x"
        )

    def _build_spinbox_row(
        self,
        parent,
        label: str,
        attr: str,
        from_: int,
        to: int,
        default: int,
    ) -> None:
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", pady=4)
        _label(row, label, fg=TEXT_SEC).pack(side="left")

        var = tk.IntVar(value=default)
        setattr(self, f"_{attr}_var", var)

        sb = ttk.Spinbox(
            row, from_=from_, to=to, textvariable=var, width=8, font=FONT_LABEL
        )
        sb.pack(side="right")

    def _build_region_selector(self, parent: tk.Frame) -> None:
        _label(parent, "지역 선택 (복수 가능)", fg=TEXT_SEC).pack(
            anchor="w", pady=(8, 4)
        )

        # 검색
        search_var = tk.StringVar()
        search_entry = ttk.Entry(parent, textvariable=search_var, font=FONT_LABEL)
        search_entry.pack(fill="x", pady=(0, 4))

        # 리스트박스 + 스크롤
        lb_frame = tk.Frame(parent, bg=PANEL_BG)
        lb_frame.pack(fill="x")

        sb = ttk.Scrollbar(lb_frame, orient="vertical")
        self._region_lb = tk.Listbox(
            lb_frame,
            selectmode="multiple",
            font=FONT_LABEL,
            height=8,
            yscrollcommand=sb.set,
            exportselection=False,
            activestyle="none",
            selectbackground=ACCENT,
            selectforeground="white",
        )
        sb.config(command=self._region_lb.yview)
        sb.pack(side="right", fill="y")
        self._region_lb.pack(side="left", fill="both", expand=True)

        for r in VALID_REGIONS:
            self._region_lb.insert("end", r)

        # 기본 전체 선택
        self._region_lb.select_set(0, "end")

        # 검색 필터
        def _filter(*_):
            q = search_var.get().strip()
            self._region_lb.delete(0, "end")
            for r in VALID_REGIONS:
                if q == "" or q in r:
                    self._region_lb.insert("end", r)
            self._region_lb.select_set(0, "end")

        search_var.trace_add("write", _filter)

        # 전체 선택/해제
        btn_row = tk.Frame(parent, bg=PANEL_BG)
        btn_row.pack(fill="x", pady=(4, 0))
        tk.Button(
            btn_row,
            text="전체 선택",
            command=lambda: self._region_lb.select_set(0, "end"),
            font=FONT_SMALL,
            relief="flat",
            bg=BORDER,
            cursor="hand2",
            padx=6,
        ).pack(side="left", padx=(0, 4))
        tk.Button(
            btn_row,
            text="전체 해제",
            command=lambda: self._region_lb.selection_clear(0, "end"),
            font=FONT_SMALL,
            relief="flat",
            bg=BORDER,
            cursor="hand2",
            padx=6,
        ).pack(side="left")

    def _build_save_path_row(self, parent: tk.Frame) -> None:
        _label(parent, "더미 데이터 CSV 저장 경로", fg=TEXT_SEC).pack(
            anchor="w", pady=(0, 4)
        )

        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x")

        self._path_var = tk.StringVar(value=os.path.join(os.getcwd(), "crime_data.csv"))
        path_entry = ttk.Entry(row, textvariable=self._path_var, font=FONT_SMALL)
        path_entry.pack(side="left", fill="x", expand=True)

        tk.Button(
            row,
            text="…",
            relief="flat",
            bg=BORDER,
            cursor="hand2",
            command=self._browse_path,
            padx=6,
        ).pack(side="right")

    # ── 오른쪽: 결과 패널 ──────────────────────
    def _build_result_panel(self, parent: tk.Frame) -> None:
        # 상태 카드
        status_card = _card(parent)
        status_card.pack(fill="x", pady=(0, 8))

        s_inner = tk.Frame(status_card, bg=PANEL_BG, padx=14, pady=10)
        s_inner.pack(fill="x")

        _label(s_inner, "처리 상태", font=FONT_BOLD).pack(anchor="w")

        self._status_var = tk.StringVar(value="대기 중")
        self._status_lbl = _label(s_inner, "", fg=TEXT_SEC)
        self._status_lbl.config(textvariable=self._status_var)
        self._status_lbl.pack(anchor="w")

        self._progress = ttk.Progressbar(s_inner, mode="indeterminate", length=300)
        self._progress.pack(fill="x", pady=(6, 0))

        # 통계 카드
        stat_card = _card(parent)
        stat_card.pack(fill="x", pady=(0, 8))

        st_inner = tk.Frame(stat_card, bg=PANEL_BG, padx=14, pady=10)
        st_inner.pack(fill="x")

        _label(st_inner, "데이터 통계", font=FONT_BOLD).pack(anchor="w", pady=(0, 6))

        self._stat_frame = tk.Frame(st_inner, bg=PANEL_BG)
        self._stat_frame.pack(fill="x")
        self._stat_labels: dict[str, tk.Label] = {}
        for key, title in [
            ("rows", "총 행수"),
            ("cols", "열 수"),
            ("crime_types", "범죄 유형"),
            ("regions", "지역 수"),
            ("year_range", "연도 범위"),
            ("crime_rate", "평균 범죄율"),
        ]:
            row = tk.Frame(self._stat_frame, bg=PANEL_BG)
            row.pack(fill="x", pady=1)
            _label(row, title, fg=TEXT_SEC, width=12, anchor="w").pack(side="left")
            lbl = _label(row, "—", fg=TEXT_PRI)
            lbl.pack(side="left")
            self._stat_labels[key] = lbl

        # 미리보기 카드
        preview_card = _card(parent)
        preview_card.pack(fill="both", expand=True)

        p_inner = tk.Frame(preview_card, bg=PANEL_BG, padx=14, pady=10)
        p_inner.pack(fill="both", expand=True)

        _label(p_inner, "미리보기 (상위 10행)", font=FONT_BOLD).pack(
            anchor="w", pady=(0, 6)
        )

        self._build_table(p_inner)

    def _build_table(self, parent: tk.Frame) -> None:
        cols = ["범죄_유형", "지역", "연도", "발생_건수", "인구수", "범죄율"]
        self._tree = ttk.Treeview(parent, columns=cols, show="headings", height=10)

        col_widths = {
            "범죄_유형": 130,
            "지역": 70,
            "연도": 55,
            "발생_건수": 80,
            "인구수": 90,
            "범죄율": 75,
        }
        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=col_widths.get(c, 80), anchor="center")

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

    # ──────────────────────────────────────────
    #  이벤트 핸들러
    # ──────────────────────────────────────────
    def _browse_path(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")],
            initialfile="crime_data.csv",
        )
        if path:
            self._path_var.set(path)

    def _on_generate(self) -> None:
        selected_indices = self._region_lb.curselection()
        regions: List[str] = [self._region_lb.get(i) for i in selected_indices]

        raw = {
            "data_count": self._data_count_var.get(),
            "year_start": self._year_start_var.get(),
            "year_end": self._year_end_var.get(),
            "region": regions,
        }

        try:
            params = get_user_input_gui(raw)
        except ValueError as e:
            messagebox.showerror("입력 오류", str(e))
            return

        errors = params.validation_errors()
        if errors:
            messagebox.showerror("유효성 오류", "\n".join(errors))
            return

        self._set_status("더미 데이터 생성 및 전처리 중…", TEXT_SEC)
        self._progress.start(12)
        self._clear_table()

        threading.Thread(target=self._run_pipeline, args=(params,), daemon=True).start()

    def _run_pipeline(self, params: GenerationParams) -> None:
        try:
            gen = DataGenerator(params)
            df_raw = gen.generate()

            vm = CrimeViewModel(callback=self._on_state_update)
            vm.process_from_df(df_raw)

            if vm.state.status == ProcessStatus.SUCCESS:
                self._df = vm.state.final_data
                self.root.after(0, self._on_success)
            else:
                self.root.after(
                    0,
                    lambda: self._on_fail(vm.state.failed_step, vm.state.error_message),
                )
        except Exception as exc:
            self.root.after(0, lambda: self._on_fail("더미 데이터 생성", str(exc)))

    def _on_state_update(self, state: CrimeState) -> None:
        if state.status == ProcessStatus.RUNNING:
            msg = f"[{len(state.completed_steps)}/4] {state.current_step} 처리 중…"
            self.root.after(0, lambda m=msg: self._set_status(m, TEXT_SEC))

    def _on_success(self) -> None:
        self._progress.stop()
        self._set_status("✅  더미 데이터 생성 완료", SUCCESS)
        self._update_stats(self._df)
        self._populate_table(self._df)
        messagebox.showinfo(
            "더미 데이터 생성 완료",
            "샘플 데이터가 생성되어 미리보기에 표시되었습니다.\n\n"
            f"CSV 저장 버튼을 누르면 아래 경로에 저장됩니다:\n{self._path_var.get().strip()}",
        )

    def _on_fail(self, step: str, message: str) -> None:
        self._progress.stop()
        self._set_status(f"❌  실패: {step}", DANGER)
        messagebox.showerror("처리 실패", f"단계: {step}\n\n{message}")

    def _on_save(self) -> None:
        if self._df is None:
            messagebox.showwarning("저장 불가", "먼저 더미 데이터를 생성하세요.")
            return
        path = self._path_var.get().strip()
        if not path:
            messagebox.showwarning("경로 없음", "저장 경로를 입력하세요.")
            return

        result = DataExporter.save_to_csv(self._df, path)
        if result:
            messagebox.showinfo("더미 CSV 저장 완료", f"저장 완료:\n{path}")
        else:
            messagebox.showerror("저장 실패", result.message)

    # ──────────────────────────────────────────
    #  UI 갱신 헬퍼
    # ──────────────────────────────────────────
    def _set_status(self, msg: str, color: str = TEXT_PRI) -> None:
        self._status_var.set(msg)
        self._status_lbl.config(fg=color)

    def _update_stats(self, df: pd.DataFrame) -> None:
        self._stat_labels["rows"].config(text=f"{len(df):,} 행")
        self._stat_labels["cols"].config(text=f"{len(df.columns)} 열")
        self._stat_labels["crime_types"].config(text=str(df["범죄_유형"].nunique()))
        self._stat_labels["regions"].config(text=str(df["지역"].nunique()))
        years = sorted(df["연도"].dropna().unique())
        self._stat_labels["year_range"].config(
            text=f"{int(years[0])} ~ {int(years[-1])}" if years else "—"
        )
        avg_rate = df["범죄율"].mean() if "범죄율" in df.columns else 0
        self._stat_labels["crime_rate"].config(text=f"{avg_rate:.2f}")

    def _clear_table(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

    def _populate_table(self, df: pd.DataFrame) -> None:
        self._clear_table()
        preview = DataExporter.preview(df, rows=10)
        for i, row in enumerate(preview["data"]):
            tag = "even" if i % 2 == 0 else "odd"
            vals = [
                row.get("범죄_유형", ""),
                row.get("지역", ""),
                row.get("연도", ""),
                f"{row.get('발생_건수', 0):,}",
                f"{row.get('인구수', 0):,}",
                f"{row.get('범죄율', 0):.2f}",
            ]
            self._tree.insert("", "end", values=vals, tags=(tag,))

        self._tree.tag_configure("even", background="#F9FAFB")
        self._tree.tag_configure("odd", background=PANEL_BG)

    # ──────────────────────────────────────────
    #  실행
    # ──────────────────────────────────────────
    def run(self) -> None:
        self.root.mainloop()
