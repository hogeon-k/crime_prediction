from __future__ import annotations

from model.excel_model import (
    CrimeState,
    ProcessStatus,
)


class CrimeView:

    def render(self, state: CrimeState) -> None:

        if state.status == ProcessStatus.SUCCESS:

            print()  # 진행중 \r 라인 정리
            print("처리 완료")

            if state.final_data is not None:
                print(f"\n[결과 미리보기 - 상위 5행]")
                print(state.final_data.head().to_string(index=False))
                print(f"\n총 {len(state.final_data):,}건 처리됨")
            else:
                print("(출력할 데이터가 없습니다)")

        elif state.status == ProcessStatus.FAILED:
            print()  # 진행중 \r 라인 정리
            print(f"   실패 단계: {state.failed_step}")
            print(f"   오류 내용: {state.error_message}")

        else:
            # 진행 중: \r로 같은 줄 덮어쓰기
            completed = len(state.completed_steps)
            print(
                f"\r[{completed}단계 완료] 현재: {state.current_step}...",
                end="",
                flush=True,
            )
