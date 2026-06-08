import numpy as np


class XGBTreeNode:
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


class XGBRegressionTree:
    """
    XGBoost용 회귀 트리
    - gradient / hessian 기반 split
    - gain이 가장 큰 분할 선택
    """

    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        reg_lambda=1.0,
        gamma=0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = None

    def fit(self, X, gradients, hessians):
        X = np.array(X, dtype=float)
        gradients = np.array(gradients, dtype=float)
        hessians = np.array(hessians, dtype=float)

        self.root = self._build_tree(
            X,
            gradients,
            hessians,
            depth=0,
        )

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)

        return np.array([self._predict_one(row, self.root) for row in X])

    def _build_tree(self, X, gradients, hessians, depth):
        n_samples = X.shape[0]

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = self._leaf_value(gradients, hessians)
            return XGBTreeNode(value=leaf_value)

        best_split = self._find_best_split(
            X,
            gradients,
            hessians,
        )

        if best_split is None:
            leaf_value = self._leaf_value(gradients, hessians)
            return XGBTreeNode(value=leaf_value)

        feature_index, threshold, gain = best_split

        if gain <= self.gamma:
            leaf_value = self._leaf_value(gradients, hessians)
            return XGBTreeNode(value=leaf_value)

        left_idx = X[:, feature_index] <= threshold
        right_idx = X[:, feature_index] > threshold

        left = self._build_tree(
            X[left_idx],
            gradients[left_idx],
            hessians[left_idx],
            depth + 1,
        )

        right = self._build_tree(
            X[right_idx],
            gradients[right_idx],
            hessians[right_idx],
            depth + 1,
        )

        return XGBTreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=left,
            right=right,
        )

    def _find_best_split(self, X, gradients, hessians):
        n_samples, n_features = X.shape

        best_gain = -float("inf")
        best_feature = None
        best_threshold = None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_idx = X[:, feature_index] <= threshold
                right_idx = X[:, feature_index] > threshold

                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                gain = self._gain(
                    gradients,
                    hessians,
                    left_idx,
                    right_idx,
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        if best_feature is None:
            return None

        return best_feature, best_threshold, best_gain

    def _gain(self, gradients, hessians, left_idx, right_idx):
        g_total = np.sum(gradients)
        h_total = np.sum(hessians)

        g_left = np.sum(gradients[left_idx])
        h_left = np.sum(hessians[left_idx])

        g_right = np.sum(gradients[right_idx])
        h_right = np.sum(hessians[right_idx])

        parent_score = self._score(g_total, h_total)
        left_score = self._score(g_left, h_left)
        right_score = self._score(g_right, h_right)

        gain = 0.5 * (left_score + right_score - parent_score)

        return gain

    def _score(self, gradient_sum, hessian_sum):
        return (gradient_sum**2) / (hessian_sum + self.reg_lambda)

    def _leaf_value(self, gradients, hessians):
        gradient_sum = np.sum(gradients)
        hessian_sum = np.sum(hessians)

        return -gradient_sum / (hessian_sum + self.reg_lambda)

    def _predict_one(self, row, node):
        if node.value is not None:
            return node.value

        if row[node.feature_index] <= node.threshold:
            return self._predict_one(row, node.left)

        return self._predict_one(row, node.right)
