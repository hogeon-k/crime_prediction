import numpy as np
import pandas as pd


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
    def prediction_diversity(y_pred, top_n=5):
        series = pd.Series(y_pred, dtype=float)
        value_counts = series.value_counts(dropna=False)
        top_values = []

        for value, count in value_counts.head(top_n).items():
            top_values.append(
                {
                    "value": float(value),
                    "count": int(count),
                    "ratio": float(count / len(series)) if len(series) else 0.0,
                }
            )

        return {
            "row_count": int(len(series)),
            "unique_count": int(series.nunique(dropna=False)),
            "unique_ratio": float(series.nunique(dropna=False) / len(series))
            if len(series)
            else 0.0,
            "min": float(series.min()) if len(series) else None,
            "max": float(series.max()) if len(series) else None,
            "mean": float(series.mean()) if len(series) else None,
            "std": float(series.std(ddof=0)) if len(series) else None,
            "top_values": top_values,
        }

    @staticmethod
    def grouped_mae(y_true, y_pred, groups):
        frame = pd.DataFrame(
            {
                "group": pd.Series(groups).astype(str).to_numpy(),
                "actual": np.array(y_true, dtype=float),
                "predicted": np.array(y_pred, dtype=float),
            }
        )
        frame["absolute_error"] = np.abs(frame["actual"] - frame["predicted"])

        return {
            group: float(value)
            for group, value in frame.groupby("group")["absolute_error"].mean().sort_values().items()
        }

    @staticmethod
    def evaluate_train_test(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        group_data: pd.DataFrame | None = None,
    ):
        train_metrics = ModelEvaluator.evaluate(y_train, y_train_pred)
        test_metrics = ModelEvaluator.evaluate(y_test, y_test_pred)
        group_metrics = {}

        if group_data is not None:
            if "지역" in group_data.columns:
                group_metrics["region_mae"] = ModelEvaluator.grouped_mae(
                    y_test, y_test_pred, group_data["지역"]
                )
            if "범죄_유형" in group_data.columns:
                group_metrics["crime_type_mae"] = ModelEvaluator.grouped_mae(
                    y_test, y_test_pred, group_data["범죄_유형"]
                )

        return {
            "metrics": test_metrics,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "prediction_diversity": ModelEvaluator.prediction_diversity(y_test_pred),
            **group_metrics,
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
                -results[name].get("test_metrics", results[name]["metrics"])["r2"],
                results[name].get("test_metrics", results[name]["metrics"])["mse"],
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
            train_metrics = item.get("train_metrics")
            test_metrics = item.get("test_metrics", item["metrics"])
            diversity = item.get("prediction_diversity", {})

            print(f"\n[{name}]")
            if train_metrics is not None:
                print(f"Train R2   : {train_metrics['r2']:.4f}")
                print(f"Train RMSE : {train_metrics['rmse']:.4f}")
                print(f"Train MAE  : {train_metrics['mae']:.4f}")
                print(f"Train MSE  : {train_metrics['mse']:.4f}")
            print(f"Test R2    : {test_metrics['r2']:.4f}")
            print(f"Test RMSE  : {test_metrics['rmse']:.4f}")
            print(f"Test MAE   : {test_metrics['mae']:.4f}")
            print(f"Test MSE   : {test_metrics['mse']:.4f}")
            if diversity:
                print(f"Unique prediction ratio: {diversity['unique_ratio']:.4f}")
                print("Top repeated predictions:")
                for row in diversity.get("top_values", []):
                    print(
                        f"  {row['value']:.6f} -> {row['count']}행 "
                        f"({row['ratio']:.2%})"
                    )
