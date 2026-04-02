import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin

class PowerBandExtractor(BaseEstimator, TransformerMixin):
    """
    Extract the average power in specific frequency bands (e.g., Mu and Beta) from EEG epochs using Fourier analysis.
    """
    def __init__(self, sfreq=160.0, bands=None):
        self.sfreq = sfreq
        if bands is None:
            self.bands = {
                "mu": (8, 12),
                "beta": (13, 30),
            }
        else:
            self.bands = bands

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Take a 3D array of shape (n_epochs, n_channels, n_times) and return a 2D array of shape (n_epochs, n_features) where n_features = n_channels * n_bands.
        Each feature corresponds to the average power in a specific frequency band for a specific channel.
        """
        n_epochs, n_channels, n_times = X.shape
        n_bands = len(self.bands)
        n_features = n_channels * n_bands
        X_features = np.zeros((n_epochs, n_features))
        print(f"Extracting spectral power features for {n_epochs} epochs...")
        for i in range(n_epochs):
            for j in range(n_channels):
                freqs, psd = welch(X[i, j, :], fs=self.sfreq, nperseg=int(self.sfreq))
                feature_idx = j * n_bands
                for band_name, (fmin, fmax) in self.bands.items():
                    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                    band_power = np.mean(psd[idx_band])
                    X_features[i, feature_idx] = band_power
                    feature_idx += 1
        return X_features
