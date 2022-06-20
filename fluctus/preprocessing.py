"""
Functions for the following operations on fmri data

mask -> extract -> interpolate -> trial average -> format -> plotting

- mask :: Union[anatomical, functional]
- extract :: extract time-series from mask
- interpolate :: generate new grid based on frequency / period
- trial average :: average trials
- format to pandas :: trials, timepoints, voxels -> pandas DataFrame
- plotting :: plotting functionality for the DataFrame based off seaborn

samples - timepoints, rows
features - voxels, columns

"""
import warnings

import numpy as np
from scipy import interpolate

from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from fluctus.bootstrap import gsbb_bootstrap_ci


class PSCScaler(BaseEstimator, TransformerMixin, _OneToOneFeatureMixin):
    """Percent Signal Change Scaler
    Standardize features by removing and dividing by the mean, and multiplying
    by 100.

    The PSC of a sample `x` is defined as:

    PSC = 100 * (x - u)/abs(u)

    where `u` is the mean.
    PSC happens independently on each feature (time-series) by computing the
    relevant stats on the training dataset.

    Parameters
    ----------
    copy : bool, default=True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
    """

    def __init__(self, *, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, "mean_"):
            del self.mean_

    def fit(self, X, y=None):
        """Compute the mean to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        self._reset()
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        """Perform PSC standardization by subtracting and dividing by the mean,
        then multiplying by 100.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        Returns
        -------
        X_tr : ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)

        signals = X.copy() if self.copy else X
        # mean_signal = signals.mean(axis=0)
        invalid_ix = np.absolute(self.mean_) < np.finfo(np.float64).eps
        signals = (signals - self.mean_) / np.absolute(self.mean_)
        signals *= 100
        if np.any(invalid_ix):
            warnings.warn(
                "psc standardization strategy is meaningless "
                "for features that have a mean of 0. "
                "These time series are set to 0."
            )
            signals[:, invalid_ix] = 0
        return signals

    def inverse_transform(self, X):
        """Scale back the data to the original representation.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)
        signals = X.copy() if self.copy else X
        signals /= 100
        signals *= np.absolute(self.mean_)
        signals += self.mean_
        return signals


class PeriodicGridTransformer(BaseEstimator, TransformerMixin):
    """
    Takes a periodic signal sampled at `sampling_in` intervals and interpolates
    it into a grid where each `period` has the same number of samples, and the
    distance between each sample approximates as good as possible the
    `target_sampling_out`.
    Discard the initial `start_offset`.
    The number of periods need not be known, but is internally computed and
    stored under the variable n_periods after the transformer has been fit.
    This also discards any remaining samples at the end of the time series if
    they don't belong in a complete period.

    Parameters
    ----------
    period: float
        The period of the underlying signal.
    sampling_in : float, default=1.0
        The sampling ratio, or time interval between two consecutive samples.
        Has to be the same units as period.
    target_sampling_out : float, default=0.1
        The target sampling rate of the output features. Note that in order to
        enforce an equal number of samples for each period, the actual sampling
        out will be as close as possible, but not necessarely identical to the
        `target_sampling_out`.
        Same units as sampling_in and period.

    Attributes
    ----------
    n_periods_ : int
        The mean value for each feature in the training set.

    sampling_out_ : float
        The actual sampling rate of the interpolated regridded signal.

    source_grid_: ndarray of shape (n_samples,)
        The original grid of the input fitted data.

    target_grid_: ndarray of shape (n_samples,)
        The grid in which to place the output samples upon calling transform.
    """

    def __init__(
        self,
        period: float,
        sampling_in: float = 1.0,
        target_sampling_out: float = 0.1,
        start_offset: float = 0.0,
    ):
        self.sampling_in = sampling_in
        self.period = period
        self.target_sampling_out = target_sampling_out
        self.start_offset = start_offset

    def _reset(self):
        """Reset internal data-dependent state of the transformer, if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, "n_periods_"):
            del self.n_periods_
            del self.source_grid_
            del self.target_grid_
            del self.sampling_out_

    def fit(self, X, y=None):
        """Compute the source and target grids, and the number of periods
        that fit in the target grid.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the source and target grids. Does not
        actually touch the data, expect for extracting its dimensions.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.

        """
        self._reset()
        self.source_grid_ = [x * self.sampling_in for x in range(X.shape[0])]
        total_time = self.source_grid_[-1]
        trials, _ = divmod(total_time - self.start_offset, self.period)
        self.n_periods_ = trials
        total_trial_time = trials * self.period / self.target_sampling_out
        sampling_points = total_trial_time // trials * trials
        self.target_grid_ = np.linspace(
            self.start_offset,
            trials * self.period + self.start_offset,
            num=int(sampling_points),
            endpoint=False,
        )
        self.sampling_out_ = self.target_grid_[1] - self.target_grid_[0]
        return self

    def transform(self, X):
        """Interpolate the input data `X` samples in source grid onto
        the target grid using cubic splines.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (source_grid_, n_features)
            The data used to interpolate along the samples axis.
        Returns
        -------
        X_tr : ndarray of shape (target_grid_, n_features)
            Transformed array.
        """
        check_is_fitted(self)
        f = interpolate.interp1d(self.source_grid_, X, axis=0, kind="cubic")
        X_out = f(self.target_grid_)
        return X_out


class TrialAveragingTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer takes a periodic signal and averages all of the trials
    (each period considered to be a trial).
    Parameters
    ----------
    n_trials: int
        The number of trials in the signal that we want to average.
    n_boots: int
        The number of bootstraps to compute during the averaging process.
    """

    def __init__(
        self,
        n_trials: int = 1,
        n_boots: int = 5000,
        ci=95,
        blocksize=25,
        bootstrap=True,
    ):
        self.n_trials = n_trials
        self.n_boots = n_boots
        self.ci = ci
        self.blocksize = blocksize
        self.bootstrap = bootstrap

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Check that we can actually divide the data by the number of trials.
        assert (X.shape[0] / self.n_trials) % 1 < np.finfo(
            np.float32
        ).eps, "X samples (shape[0]) should be a multiple of n_trials."

        if self.bootstrap:
            period = X.shape[0] // self.n_trials
            self.ci_low_, self.ci_high_ = gsbb_bootstrap_ci(
                X,
                int(period),
                blocksize=self.blocksize,
                n_boots=self.n_boots,
                level=self.ci,
            )
            self.ci_low_ = self.transform(self.ci_low_)
            self.ci_high_ = self.transform(self.ci_high_)
        return self

    def transform(self, X):
        """Splits and averages signals according to trials."""
        return np.dstack(np.split(X, self.n_trials)).mean(-1)


class FFTTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer performs an FFT on the time-series, and returns its power
    spectrum. The spectrum is normalized such that its DC component is
    equivalent to the mean of the transformed series. All other values represent
    amplitudes of each frequency in the same units as the DC.

    Parameters
    ----------
    sampling_rate: float, the sampling rate of each sample.

    Attributes
    ----------
    freqs_ : int
        The grid of frequencies from the transformation.

    Extra methods
    ----------
    closest_freq_index(frequency: float)
        Returns the closest index in the freqs_ attribute that corresponds to the
        given `frequency` argument.

    """

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def _reset(self):
        """Reset internal data-dependent state of the transformer, if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, "freqs_"):
            del self.freqs_

    def fit(self, X, y=None):
        """Compute the grid of frequencies upon which to place the transform.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the source and target grids. Does not
        actually touch the data, expect for extracting its dimensions.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.

        """
        self._reset()
        self.freqs_ = np.fft.rfftfreq(X.shape[0], self.sampling_rate)
        return self

    def closest_freq_index(self, frequency):
        """Returns the closest index to the desired frequency in fitted
        frequency domain."""
        check_is_fitted(self)
        return np.argmin(np.abs(self.freqs_ - frequency))

    def transform(self, X):
        """Returns power spectrum of signals."""
        check_is_fitted(self)
        power_spectra = np.abs(np.fft.rfft(X, axis=0) / X.shape[0])
        idx = np.argsort(self.freqs_)
        return power_spectra[idx, :].squeeze()


class FeatureAverager(BaseEstimator, TransformerMixin):
    """
    This transformer averages multiple features (in the sklearn lingo).
    In the case where features represent distinct timeseries, then this
    transformer can be used to average them.

    Notes
    -----
    This transformer is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.

    It can be used with ColumnTransformer and the feature extraction package to
    average signals from distinct groups/clusters/classes.
    """

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        """Averages input features (columns) and returns them as a 2D array."""
        return X.mean(axis=-1)[:, np.newaxis]
