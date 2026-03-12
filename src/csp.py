import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg=1e-6):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError("CSP expects a 3D array of shape (epochs, channels, time).")

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, found {classes.tolist()}.")

        if self.n_components < 2:
            raise ValueError("CSP requires at least 2 components.")

        class_a, class_b = classes
        cov_a = self._mean_normalized_covariance(X[y == class_a])
        cov_b = self._mean_normalized_covariance(X[y == class_b])

        composite_cov = cov_a + cov_b
        eigvals, eigvecs = eigh(composite_cov)
        eigvals = np.maximum(eigvals, self.reg)

        whitening = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        whitened_cov_a = whitening @ cov_a @ whitening.T

        csp_eigvals, csp_eigvecs = eigh(whitened_cov_a)
        order = np.argsort(csp_eigvals)[::-1]
        csp_eigvecs = csp_eigvecs[:, order]

        filters = csp_eigvecs.T @ whitening
        selected_indices = self._select_component_indices(filters.shape[0], self.n_components)

        self.classes_ = classes
        self.filters_ = filters[selected_indices]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("CSP expects a 3D array of shape (epochs, channels, time).")

        projected = np.einsum("fc,ect->eft", self.filters_, X)
        variances = np.var(projected, axis=2)
        normalized_variances = variances / np.sum(variances, axis=1, keepdims=True)
        return np.log(normalized_variances)

    def _mean_normalized_covariance(self, epochs):
        covariances = [self._normalized_covariance(epoch) for epoch in epochs]
        return np.mean(covariances, axis=0)

    def _normalized_covariance(self, epoch):
        covariance = epoch @ epoch.T
        trace = np.trace(covariance)
        if trace <= 0:
            raise ValueError("Encountered a non-positive covariance trace while fitting CSP.")
        covariance = covariance / trace
        covariance += self.reg * np.eye(covariance.shape[0])
        return covariance

    def _select_component_indices(self, total_components, n_components):
        n_components = min(n_components, total_components)
        indices = []
        left = 0
        right = total_components - 1

        while len(indices) < n_components and left <= right:
            indices.append(left)
            if len(indices) < n_components and right != left:
                indices.append(right)
            left += 1
            right -= 1

        return indices
