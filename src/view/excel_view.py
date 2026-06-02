from model.excel_model import ExcelState, ProcessStatus

STEP_ORDER = ["파일 로드", "컬럼 통일", "결측치 처리", "타입 변환"]


class ExcelView:
    """
    MVVM - View (CLI)
    ExcelState 만 수신하며 출력만 담당 — 로직 없음
    """

    def render(self, state: ExcelState) -> None:
        match state.status:
            case ProcessStatus.RUNNING:
                self._render_progress(state)
            case ProcessStatus.SUCCESS:
                self._render_success(state)
            case ProcessStatus.FAILED:
                self._render_failure(state)

    # ── private ────────────────────────────────────
    def _render_progress(self, state: ExcelState) -> None:
        print(f"\r⏳  {state.current_step} ...", end="", flush=True)

    def _render_success(self, state: ExcelState) -> None:
        print("\n")
        print("╔══════════════════════════════════════╗")
        print("║          ✅  처리 완료                ║")
        print("╚══════════════════════════════════════╝")
        for step in state.completed_steps:
            print(f"  ✓  {step}")
        if state.final_data is not None:
            df = state.final_data
            print(f"\n  행 수  : {len(df):,}")
            print(f"  컬럼   : {list(df.columns)}")
            print(f"  dtypes :\n{df.dtypes.to_string()}")
            print(f"\n── 상위 5행 ──")
            print(df.head(5).to_string(index=False))

    def _render_failure(self, state: ExcelState) -> None:
        print("\n")
        print("╔══════════════════════════════════════╗")
        print("║          ❌  처리 실패                ║")
        print("╚══════════════════════════════════════╝")
        reached = len(state.completed_steps)
        for step in state.completed_steps:
            print(f"  ✓  {step}")
        print(f"  ✗  {state.failed_step}  ← 여기서 중단")
        for step in STEP_ORDER[reached + 1 :]:
            print(f"  ─  {step}  (건너뜀)")
        print(f"\n  오류: {state.error_message}")
