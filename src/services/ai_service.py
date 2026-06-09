from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai.predict import (
    DEFAULT_MODEL_INFO_PATH,
    DEFAULT_MODEL_PATH,
    load_best_model,
    predict_from_file,
    predict_one,
)


class AIService:
    """저장된 AI 모델 로드와 예측 실행을 담당하는 service 계층."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        model_info_path: str | Path = DEFAULT_MODEL_INFO_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_info_path = Path(model_info_path)

    def model_exists(self) -> bool:
        return self.model_path.exists() and self.model_info_path.exists()

    def load_model(self):
        """SHA256 검증을 포함해 저장된 모델을 로드한다."""
        return load_best_model(self.model_path, info_path=self.model_info_path)

    def predict_one(
        self,
        year: int,
        region: str,
        crime_type: str,
        population: int,
    ) -> float:
        model = self.load_model()
        return predict_one(
            year=year,
            region=region,
            crime_type=crime_type,
            population=population,
            model=model,
        )

    def predict_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_year: int,
        debug: bool = False,
        debug_printer=print,
    ) -> pd.DataFrame:
        return predict_from_file(
            input_path,
            output_path,
            target_year=target_year,
            debug=debug,
            debug_printer=debug_printer,
        )

    @staticmethod
    def format_prediction_error(message: str) -> str:
        if "best_model.pkl" in message:
            return "먼저 src/ai/train.py를 실행해 모델을 생성하거나 release asset에서 모델 파일을 내려받으세요."

        if "예측에 필요한 컬럼" in message:
            return (
                f"{message}\n\n"
                "필수 컬럼: 연도, 지역, 범죄_유형, 인구수\n"
                "예측 샘플 파일 생성 기능으로 입력 형식을 확인할 수 있습니다."
            )

        if "예측 대상 연도" in message:
            return message

        if "학습 데이터에 없는 예측 입력값" in message:
            return (
                f"{message}\n\n"
                "입력 파일의 지역/범죄_유형 값을 학습 데이터와 같은 이름으로 맞춰주세요.\n"
                "학습에 전혀 없는 값은 예측할 수 없습니다."
            )

        if "모델 파일 해시" in message or "model_sha256" in message:
            return (
                f"{message}\n\n"
                "모델 파일과 model_info.json의 무결성 정보를 확인하세요."
            )

        return message
