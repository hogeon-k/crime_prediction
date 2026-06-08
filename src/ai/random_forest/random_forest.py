import numpy as np

from ai.random_forest.decision_tree import DecisionTreeRegressor


class RandomForestRegressorModel:
    """
    직접 구현한 Random Forest Regressor

    핵심:
    - bootstrap sampling
    - 여러 DecisionTree 학습
    - 예측값 평균
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        np.random.seed(self.random_state)

        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)

        if not self.trees:
            raise ValueError(
                "모델이 아직 학습되지 않았습니다. fit()을 먼저 실행하세요."
            )

        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        return np.mean(tree_predictions, axis=0)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]

        indices = np.random.choice(
            n_samples,
            size=n_samples,
            replace=True,
        )

        return X[indices], y[indices]

    def get_state(self):
        """
        pkl 저장용 상태 반환
        """

        return {
            "model_type": "random_forest_regressor",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "trees": self.trees,
        }

    def load_state(self, state):
        """
        저장된 상태 복원
        """

        self.n_estimators = state["n_estimators"]
        self.max_depth = state["max_depth"]
        self.min_samples_split = state["min_samples_split"]
        self.min_samples_leaf = state.get("min_samples_leaf", 1)
        self.max_features = state["max_features"]
        self.random_state = state["random_state"]
        self.trees = state["trees"]

        return self
