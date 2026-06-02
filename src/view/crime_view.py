from model.excel_model import (
    CrimeState,
    ProcessStatus,
)


class CrimeView:

    def render(
        self,
        state: CrimeState,
    ):

        if state.status == ProcessStatus.SUCCESS:

            print("\n처리 완료")

            print(state.final_data.head())

        elif state.status == ProcessStatus.FAILED:

            print(f"\n실패: " f"{state.error_message}")

        else:
            print(f"\r[진행중] {state.current_step}...", end="", flush=True)

        # SUCCESS 케이스에 줄바꿈 추가
        if state.status == ProcessStatus.SUCCESS:
            print()  # 이전 \r 라인 정리
            print("\n✅ 처리 완료")
