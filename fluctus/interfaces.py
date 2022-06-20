"""
This provides a high-level API for dealing with oscillations
The idea is that a user provides either a time-series or collection of
time-series, and we provide objects for conveniently manipulating them.

By chaining transforms, one can easily generate any analysis of interest.
"""
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiMasker
from scipy.signal import butter, sosfiltfilt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

from fluctus import preprocessing


# Helpers
def get_ntrials(start_offset, period, tr, nvols):
    source_grid_ = [x * tr for x in range(nvols)]
    total_time = source_grid_[-1]
    trials, _ = divmod(total_time - start_offset, period)
    return trials


def get_offset(stimulus_offset=14, period=10):
    offset = stimulus_offset
    # 39.999 just so we accept 54 as valid. Python doesn't have a do while loop.
    while offset <= (stimulus_offset + 39.999):  # 40 seconds after start of stim
        offset += period
    return offset


@dataclass
class Oscillation:
    """
    A class to represent an oscillation and offer methods to trial-average, voxel-average,
    PSC normalization, and FFT.
    Also offers a method to extract the time-series of a given mask and/or label from nifti.
    """

    tr: float
    period: float
    data: np.array
    stimulus_offset: float = 14
    labels: Optional[list] = None

    def __post_init__(self):
        self.transformed_data = self.data
        self.transformation_chain = []
        self.sampling_rate = self.tr
        self.grid = np.arange(self.data.shape[0]) * self.sampling_rate
        self.offset = get_offset(self.stimulus_offset, self.period)
        self.n_trials = get_ntrials(
            self.offset, self.period, self.tr, self.data.shape[0]
        )
        self.emin, self.emax = None, None
        self.ids = [""]

    def reset(self):
        self.transformed_data = self.data
        self.transformation_chain = []
        self.sampling_rate = self.tr
        self.grid = np.arange(self.data.shape[0]) * self.sampling_rate
        self.emin, self.emax = None, None

    def clear_labels(self):
        self.labels = None

    def _transform(self, transformer, id: str):
        self.transformed_data = transformer.fit_transform(self.transformed_data)
        self.transformation_chain.append(id)
        return self

    def average(self):
        if self.labels is None:
            transformer = preprocessing.FeatureAverager()
            self.ids = [""]
        else:
            label_ids = set(self.labels)
            transforms = []
            for id in label_ids:
                transforms.append(
                    (
                        id,
                        preprocessing.FeatureAverager(),
                        np.where(np.array(self.labels) == id)[0],
                    )
                )
            transformer = ColumnTransformer(transforms)
            self.ids = label_ids
        return self._transform(transformer, "Label Average")

    def psc(self):
        transformer = preprocessing.PSCScaler()
        return self._transform(transformer, "PSC")

    def trial_average(self, bootstrap: bool = False):
        transformer = preprocessing.TrialAveragingTransformer(
            n_trials=self.n_trials, bootstrap=bootstrap
        )
        transformed = self._transform(transformer, "Trial Average")
        if bootstrap:
            self.emin, self.emax = transformer.ci_low_, transformer.ci_high_
        self.grid = self.grid[: self.transformed_data.shape[0]]
        return transformed

    def interp(self, target_sampling_out: float = 0.1):
        transformer = preprocessing.PeriodicGridTransformer(
            period=self.period,
            sampling_in=self.tr,
            target_sampling_out=target_sampling_out,
            start_offset=self.offset,
        )
        transformed = self._transform(transformer, "Crop and Interpolate")
        self.grid = transformer.target_grid_ - self.offset
        return transformed

    def fft(self):
        transformer = preprocessing.FFTTransformer(self.sampling_rate)
        return self._transform(transformer, "FFT")

    def preprocess(self):
        self.reset()
        self = self.interp().psc().average().trial_average(bootstrap=True)
        return self.transformed_data.squeeze()

    @classmethod
    def from_nifti(
        cls, mask: str, data: str, period: float, labels=None, stimulus_offset=14
    ):
        masker = NiftiMasker(mask_img=mask, verbose=True)
        dat = nib.load(data)
        ts = masker.fit_transform(data)
        init = cls(
            tr=dat.header["pixdim"][4],
            period=period,
            data=ts,
            labels=labels,
            stimulus_offset=stimulus_offset,
        )
        init._masker = masker
        init._filename = data
        init._maskname = mask
        return init

    @property
    def phase(self):
        return self.grid[self.transformed_data.argmin(0)]

    @property
    def amplitude(self):
        min_max = MinMaxScaler()
        min_max.fit_transform(self.transformed_data)
        ymin, ymax = min_max.data_min_, min_max.data_max_
        xmin, xmax = (
            self.grid[self.transformed_data.argmin(0)],
            self.grid[self.transformed_data.argmax(0)],
        )
        return ymax - ymin

    def plot(self, plotci: bool = True):
        fig, ax = plt.subplots(dpi=150)
        # May have to deal with multiple curves, so...
        for i, label in enumerate(self.ids):
            ax.plot(
                self.grid[: self.transformed_data.size],
                self.transformed_data[:, i],
                label=label,
            )
            if self.emin is not None and plotci:
                ax.fill_between(
                    self.grid[: self.transformed_data.size],
                    self.emin[:, i],
                    self.emax[:, i],
                    alpha=0.4,
                )

        if "PSC" in self.transformation_chain:
            ylabel = "BOLD % Amplitude"
        else:
            ylabel = "Amplitude"

        ax.set(ylabel=ylabel, xlabel="Time (s)")

        if len(self.ids) > 1:
            ax.legend()

        return fig, ax


## Below are some helper functions to build the confidence index from Regan.


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def ratio(trace):
    aplus = np.sum(np.where(trace > 0, trace, 0))
    aminus = np.sum(np.where(trace < 0, trace, 0))
    ratio = np.abs((aplus + aminus) / (aplus - aminus))
    return ratio


def confidence_index(trace_s, trace_c):
    ratio_s = ratio(trace_s)
    ratio_c = ratio(trace_c)
    A = np.sum(trace_s)
    B = np.sum(trace_c)
    amp = np.hypot(A, B)
    conf = (A ** 2 * ratio_s + B ** 2 * ratio_c) / amp ** 2
    return conf


# Since the stimulus starts at 0 and increases, the phase needs an extra np.pi
def phase_estimate(trace_s, trace_c):
    ph = np.arctan2(np.mean(trace_s), np.mean(trace_c))
    return ph


def amplitude_estimate(trace_s, trace_c):
    return np.hypot(np.mean(trace_s), np.mean(trace_c)) / 2


def make_traces(signals, frequency, tr, offset=14, window=50):
    sos = butter(4, frequency / 2, output="sos", fs=1 / tr)
    t = np.array([x * tr for x in range(signals.shape[0])])
    s = np.sin(
        2 * np.pi * frequency * (t - offset + tr / 2)
    )  # adding tr/2 to account for slice time
    c = np.cos(2 * np.pi * frequency * (t - offset + tr / 2))

    multiplier_s = signals * s
    multiplier_c = signals * c
    trace_s = moving_average(sosfiltfilt(sos, multiplier_s), window)[window:-window]
    trace_c = moving_average(sosfiltfilt(sos, multiplier_c), window)[window:-window]
    return trace_s, trace_c


# SLOW BUT NOT BUGGY
def confidence_and_estimates(signals, frequency, tr):
    traces = [
        make_traces(signals[:, x], frequency, tr) for x in range(signals.shape[-1])
    ]
    phase_delays = [
        np.rad2deg(phase_estimate(*trace) + np.pi) * (1 / frequency) / 360
        for trace in traces
    ]
    amps = [amplitude_estimate(*trace) for trace in traces]
    confs = [confidence_index(*trace) for trace in traces]
    return np.array(phase_delays), np.array(amps), np.array(confs)
