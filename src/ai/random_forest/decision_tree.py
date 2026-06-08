import numpy as np


class TreeNode:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressor:
    """
    직접 구현한 회귀용 Decision Tree
    - MSE 감소 기준으로 split
    - leaf 값은 y 평균
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)

        return np.array([self._predict_one(row, self.root) for row in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return TreeNode(value=np.mean(y))

        feature_indices = self._select_features(n_features)

        best_feature, best_threshold, best_score = self._best_split(
            X,
            y,
            feature_indices,
        )

        if best_feature is None:
            return TreeNode(value=np.mean(y))

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        left = self._build_tree(
            X[left_idx],
            y[left_idx],
            depth + 1,
        )

        right = self._build_tree(
            X[right_idx],
            y[right_idx],
            depth + 1,
        )

        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left,
            right=right,
        )

    def _best_split(self, X, y, feature_indices):
        best_feature = None
        best_threshold = None
        best_score = float("inf")

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_idx = X[:, feature_index] <= threshold
                right_idx = X[:, feature_index] > threshold

                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                score = self._weighted_mse(
                    y[left_idx],
                    y[right_idx],
                )

                if score < best_score:
                    best_score = score
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_score

    def _weighted_mse(self, left_y, right_y):
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right

        left_mse = np.mean((left_y - np.mean(left_y)) ** 2)
        right_mse = np.mean((right_y - np.mean(right_y)) ** 2)

        return (n_left / n_total) * left_mse + (n_right / n_total) * right_mse

    def _select_features(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        max_features = min(self.max_features, n_features)

        return np.random.choice(
            n_features,
            size=max_features,
            replace=False,
        )

    def _predict_one(self, row, node):
        if node.value is not None:
            return node.value

        if row[node.feature_index] <= node.threshold:
            return self._predict_one(row, node.left)

        return self._predict_one(row, node.right)
