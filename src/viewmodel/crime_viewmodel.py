from model.excel_model import (
    CrimeState,
    ProcessStatus,
)

from services.crime_service import CrimeService


class CrimeViewModel:

    def __init__(self, callback):

        self._callback = callback

        self._service = CrimeService()

        self.state = CrimeState()

    def process(
        self,
        crime_files,
        pop_files,
    ):

        steps = [
            (
                "데이터 병합",
                lambda _: self._service.load_and_merge(crime_files, pop_files),
            ),
            (
                "검증",
                self._service.validate,
            ),
            (
                "결측치 처리",
                self._service.handle_missing,
            ),
            (
                "타입 변환",
                self._service.convert_types,
            ),
        ]

        df = None

        for name, fn in steps:

            self.state.current_step = name

            self._callback(self.state)

            result = fn(df)

            if not result.success:

                self.state.status = ProcessStatus.FAILED

                self.state.failed_step = name

                self.state.error_message = result.message

                self._callback(self.state)

                return

            df = result.data

            self.state.completed_steps.append(name)

        self.state.status = ProcessStatus.SUCCESS

        self.state.final_data = df

        self._callback(self.state)
