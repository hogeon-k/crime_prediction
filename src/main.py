"""
진입점 — View 와 ViewModel 을 연결하는 유일한 곳

비동기 (Daemon thread):
    vm.process(path)          # 즉시 반환, 콜백으로 상태 수신
    worker.join()             # 완료 대기 (CLI 용)

동기 (테스트 / 단순 CLI):
    state = vm.process_sync(path)
"""

import sys
import threading

from view.excel_view import ExcelView
from viewmodel.excel_viewmodel import ExcelViewModel


def main(file_path: str) -> None:
    view = ExcelView()

    # ── View ↔ ViewModel 바인딩 (콜백 방식) ──
    vm = ExcelViewModel(on_state_changed=view.render)

    # ── 비동기 실행 (Daemon thread) ──────────
    # process() 는 즉시 반환 — 파이프라인은 Sub thread 에서 실행
    vm.process(file_path)

    # CLI 에서는 Sub thread 종료를 기다려야 프로세스가 끝나지 않음
    # (GUI 환경에서는 이 join 이 필요 없음)
    for t in threading.enumerate():
        if t.name == "ExcelPipeline":
            t.join()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data.xlsx"
    main(path)
