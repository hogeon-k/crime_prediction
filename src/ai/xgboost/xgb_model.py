import numpy as np

from ai.xgboost.loss import SquaredErrorLoss
from ai.xgboost.tree import XGBRegressionTree


class XGBoostRegressorModel:
    """
    직접 구현한 간소화 XGBoost Regressor

    핵심:
    - 초기 예측값은 y 평균
    - gradient / hessian 계산
    - gain 기반 regression tree 학습
    - learning_rate로 예측값 갱신
    """

    def __init__(
        self,
        n_estimators=30,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        reg_lambda=1.0,
        gamma=0.0,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma

        self.base_prediction = None
        self.trees = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        self.base_prediction = np.mean(y)
        y_pred = np.full(
            shape=y.shape,
            fill_value=self.base_prediction,
            dtype=float,
        )

        self.trees = []

        for _ in range(self.n_estimators):
            gradients = SquaredErrorLoss.gradient(
                y,
                y_pred,
            )

            hessians = SquaredErrorLoss.hessian(
                y,
                y_pred,
            )

            tree = XGBRegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
            )

            tree.fit(
                X,
                gradients,
                hessians,
            )

            update = tree.predict(X)

            y_pred += self.learning_rate * update

            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)

        if self.base_prediction is None:
            raise ValueError(
                "모델이 아직 학습되지 않았습니다. fit()을 먼저 실행하세요."
            )

        y_pred = np.full(
            shape=(X.shape[0],),
            fill_value=self.base_prediction,
            dtype=float,
        )

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def get_state(self):
        """
        pkl 저장용 상태 반환
        """

        return {
            "model_type": "xgboost_regressor",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "base_prediction": self.base_prediction,
            "trees": self.trees,
        }

    def load_state(self, state):
        """
        저장된 상태 복원
        """

        self.n_estimators = state["n_estimators"]
        self.learning_rate = state["learning_rate"]
        self.max_depth = state["max_depth"]
        self.min_samples_split = state["min_samples_split"]
        self.reg_lambda = state["reg_lambda"]
        self.gamma = state["gamma"]
        self.base_prediction = state["base_prediction"]
        self.trees = state["trees"]

        return self
