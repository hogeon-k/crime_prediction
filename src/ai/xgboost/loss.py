import numpy as np


class SquaredErrorLoss:
    """
    회귀용 손실 함수: MSE 기반
    L = 1/2 * (y - y_pred)^2
    """

    @staticmethod
    def gradient(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def hessian(y_true, y_pred):
        return np.ones_like(y_true, dtype=float)
