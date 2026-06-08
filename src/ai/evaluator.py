import numpy as np


class ModelEvaluator:
    """
    Linear / RandomForest / XGBoost 공통 평가 클래스
    """

    @staticmethod
    def mse(y_true, y_pred):
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(ModelEvaluator.mse(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2(y_true, y_pred):
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    @staticmethod
    def evaluate(y_true, y_pred):
        """
        단일 모델 평가
        """

        return {
            "mse": ModelEvaluator.mse(y_true, y_pred),
            "rmse": ModelEvaluator.rmse(y_true, y_pred),
            "mae": ModelEvaluator.mae(y_true, y_pred),
            "r2": ModelEvaluator.r2(y_true, y_pred),
        }

    @staticmethod
    def compare(results):
        """
        여러 모델 비교

        기준:
        R2가 가장 높은 모델
        """

        if not results:
            raise ValueError("비교할 모델 결과가 없습니다.")

        best_name = min(
            results,
            key=lambda name: (
                -results[name]["metrics"]["r2"],
                results[name]["metrics"]["mse"],
            ),
        )

        return best_name, results[best_name]

    @staticmethod
    def print_report(results):
        """
        모델별 평가 결과 출력
        """

        print("\n========== 모델 평가 결과 ==========")

        for name, item in results.items():
            metrics = item["metrics"]

            print(f"\n[{name}]")
            print(f"MSE  : {metrics['mse']:.4f}")
            print(f"RMSE : {metrics['rmse']:.4f}")
            print(f"MAE  : {metrics['mae']:.4f}")
            print(f"R2   : {metrics['r2']:.4f}")

        best_name, _ = ModelEvaluator.compare(results)

        print("\n========== 최종 선택 모델 ==========")
        print(f"Best Model: {best_name}")
