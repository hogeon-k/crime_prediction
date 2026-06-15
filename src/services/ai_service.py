from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ai.predict import (
    DEFAULT_MODEL_INFO_PATH,
    DEFAULT_MODEL_PATH,
    load_best_model,
    predict_from_file,
    predict_recursive_from_file,
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
        self.last_inference_seconds: float | None = None
        self.last_cpu_usage_percent: float | None = None

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
        previous_incidents: float | None = None,
        previous_rate: float | None = None,
    ) -> float:
        model = self.load_model()
        started_at = time.perf_counter()
        cpu_started_at = time.process_time()
        result = predict_one(
            year=year,
            region=region,
            crime_type=crime_type,
            population=population,
            previous_incidents=previous_incidents,
            previous_rate=previous_rate,
            model=model,
        )
        self._record_resource_usage(started_at, cpu_started_at)
        return result

    def predict_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_year: int,
        debug: bool = False,
        debug_printer=print,
    ) -> pd.DataFrame:
        model = self.load_model()
        started_at = time.perf_counter()
        cpu_started_at = time.process_time()
        result = predict_from_file(
            input_path,
            output_path,
            target_year=target_year,
            model=model,
            allow_existing_target_year=True,
            debug=debug,
            debug_printer=debug_printer,
        )
        self._record_resource_usage(started_at, cpu_started_at)
        return result

    def predict_recursive_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        start_year: int,
        end_year: int,
    ) -> dict[int, pd.DataFrame]:
        model = self.load_model()
        started_at = time.perf_counter()
        cpu_started_at = time.process_time()
        result = predict_recursive_from_file(
            input_path,
            output_dir,
            start_year=start_year,
            end_year=end_year,
            model=model,
        )
        self._record_resource_usage(started_at, cpu_started_at)
        return result

    def _record_resource_usage(self, started_at: float, cpu_started_at: float) -> None:
        wall_seconds = max(time.perf_counter() - started_at, 0.0)
        cpu_seconds = max(time.process_time() - cpu_started_at, 0.0)
        self.last_inference_seconds = wall_seconds
        self.last_cpu_usage_percent = (
            min(100.0, cpu_seconds / wall_seconds * 100.0)
            if wall_seconds > 0
            else 0.0
        )

    def get_model_performance_rows(self) -> list[dict[str, float | str | None]]:
        default_rows = [
            {
                "model": "최종 저장 모델",
                "mse": None,
                "rmse": None,
                "mae": None,
                "r2": None,
                "train_r2": None,
                "selection_reason": "",
                "walk_forward_mean_r2": None,
                "walk_forward_mean_rmse": None,
                "walk_forward_mean_mae": None,
                "walk_forward_mean_mse": None,
                "inference_seconds": None,
                "cpu_usage_percent": None,
            },
        ]
        if not self.model_info_path.exists():
            return default_rows

        try:
            with self.model_info_path.open("r", encoding="utf-8") as file:
                model_info = json.load(file)
        except Exception:
            return default_rows

        best_model = str(model_info.get("best_model", ""))
        metrics = model_info.get("metrics", {})
        walk_forward = model_info.get("walk_forward_averages", {}).get(best_model, {})
        selection_reason = " ".join(model_info.get("selection_reason", []))

        return [
            {
                **default_rows[0],
                "model": model_info.get("final_saved_model") or best_model or "linear",
                "mse": metrics.get("mse"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "r2": metrics.get("r2"),
                "train_r2": metrics.get("train_r2"),
                "selection_reason": selection_reason,
                "walk_forward_mean_r2": walk_forward.get("mean_r2"),
                "walk_forward_mean_rmse": walk_forward.get("mean_rmse"),
                "walk_forward_mean_mae": walk_forward.get("mean_mae"),
                "walk_forward_mean_mse": walk_forward.get("mean_mse"),
                "inference_seconds": self.last_inference_seconds,
                "cpu_usage_percent": self.last_cpu_usage_percent,
            }
        ]

    def get_model_performance_summary(self) -> dict[str, float | str | None]:
        if not self.model_info_path.exists():
            return {
                "message": "모델 성능 정보 파일을 찾을 수 없습니다. 예측은 가능하지만 성능 정보는 표시하지 않습니다.",
                "model": None,
                "r2": None,
                "train_r2": None,
                "rmse": None,
                "mae": None,
                "mse": None,
                "selection_reason": "",
                "walk_forward_mean_r2": None,
                "walk_forward_mean_rmse": None,
                "walk_forward_mean_mae": None,
                "walk_forward_mean_mse": None,
                "inference_seconds": self.last_inference_seconds,
                "cpu_usage_percent": self.last_cpu_usage_percent,
            }

        try:
            with self.model_info_path.open("r", encoding="utf-8") as file:
                model_info = json.load(file)
        except Exception:
            return {
                "message": "모델 성능 정보 파일을 읽을 수 없습니다. 예측은 가능하지만 성능 정보는 표시하지 않습니다.",
                "model": None,
                "r2": None,
                "train_r2": None,
                "rmse": None,
                "mae": None,
                "mse": None,
                "selection_reason": "",
                "walk_forward_mean_r2": None,
                "walk_forward_mean_rmse": None,
                "walk_forward_mean_mae": None,
                "walk_forward_mean_mse": None,
                "inference_seconds": self.last_inference_seconds,
                "cpu_usage_percent": self.last_cpu_usage_percent,
            }

        metrics = model_info.get("metrics", {})
        best_model = str(model_info.get("final_saved_model") or model_info.get("best_model") or "")
        walk_forward = model_info.get("walk_forward_averages", {}).get(best_model, {})
        return {
            "message": "",
            "model": best_model or model_info.get("best_model"),
            "r2": metrics.get("r2"),
            "train_r2": metrics.get("train_r2"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mse": metrics.get("mse"),
            "selection_reason": " ".join(model_info.get("selection_reason", [])),
            "walk_forward_mean_r2": walk_forward.get("mean_r2"),
            "walk_forward_mean_rmse": walk_forward.get("mean_rmse"),
            "walk_forward_mean_mae": walk_forward.get("mean_mae"),
            "walk_forward_mean_mse": walk_forward.get("mean_mse"),
            "inference_seconds": self.last_inference_seconds,
            "cpu_usage_percent": self.last_cpu_usage_percent,
        }

    @staticmethod
    def format_prediction_error(message: str) -> str:
        if (
            "best_model.pkl" in message
            or "model_info.json" in message
            or "모델 메타데이터 파일이 없습니다" in message
        ):
            return "모델 파일을 찾을 수 없습니다. 먼저 학습을 실행하거나 models/best_model.pkl을 확인해주세요."

        if "예측에 필요한 컬럼" in message:
            missing = message.split(":", 1)[-1].strip() if ":" in message else ""
            return (
                f"필수 컬럼 누락: {missing}\n\n"
                "필수 컬럼: 지역, 범죄_유형, 연도, 인구수\n"
                "예측 샘플 파일 data/sample_prediction_input.xlsx 형식과 동일하게 작성하세요."
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
