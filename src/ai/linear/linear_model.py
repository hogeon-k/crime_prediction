import numpy as np


class LinearRegressionModel:
    """
    직접 구현한 선형 회귀 모델

    역할:
    - fit()      : 학습
    - predict()  : 예측
    - get_state(): pkl 저장용 상태 반환
    - load_state(): 저장된 상태 복원
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = 0.0

        self.train_losses = []

    def fit(self, X, y):
        """
        X: feature 데이터, shape = (행 개수, feature 개수)
        y: 정답 데이터, shape = (행 개수,)
        """

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = self.predict(X)

            error = y_pred - y

            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            mse = np.mean(error**2)
            self.train_losses.append(mse)

        return self

    def predict(self, X):
        """
        X를 입력받아 예측값 반환
        """

        X = np.array(X, dtype=float)

        if self.weights is None:
            raise ValueError(
                "모델이 아직 학습되지 않았습니다. fit()을 먼저 실행하세요."
            )

        return np.dot(X, self.weights) + self.bias

    def get_state(self):
        """
        모델 저장용 상태 반환
        """

        return {
            "model_type": "linear_regression",
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "weights": self.weights,
            "bias": self.bias,
            "train_losses": self.train_losses,
        }

    def load_state(self, state):
        """
        저장된 모델 상태 복원
        """

        self.learning_rate = state["learning_rate"]
        self.epochs = state["epochs"]
        self.weights = state["weights"]
        self.bias = state["bias"]
        self.train_losses = state.get("train_losses", [])

        return self
