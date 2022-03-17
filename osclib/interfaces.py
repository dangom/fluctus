"""
This provides a high-level API for dealing with oscillations
The idea is that a user provides either a time-series or collection of
time-series, and we provide objects for conveniently manipulating them.

By chaining transforms, one can easily generate any analysis of interest.
"""
from typing import Optional
from dataclasses import dataclass
import numpy as np
from osclib import transformers
from sklearn.compose import ColumnTransformer
import nibabel as nib
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt

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

    def reset(self):
        self.transformed_data = self.data
        self.transformation_chain = []
        self.sampling_rate = self.tr
        self.grid = np.arange(self.data.shape[0]) * self.sampling_rate
        self.emin, self.emax = None, None

    def _transform(self, transformer, id: str):
        self.transformed_data = transformer.fit_transform(self.transformed_data)
        self.transformation_chain.append(id)
        return self

    def average(self):
        if self.labels is None:
            transformer = transformers.FeatureAverager()
            self.ids = "All"
        else:
            label_ids = set(self.labels)
            transforms = []
            for id in label_ids:
                transforms.append(
                    (id, transformers.FeatureAverager(), np.where(self.labels == id)[0])
                )
            transformer = ColumnTransformer(transforms)
            self.ids = label_ids
        return self._transform(transformer, "Label Average")

    def psc(self):
        transformer = transformers.PSCScaler()
        return self._transform(transformer, "PSC")

    def trial_average(self, bootstrap: bool = False):
        transformer = transformers.TrialAveragingTransformer(
            n_trials=self.n_trials, bootstrap=bootstrap
        )
        transformed = self._transform(transformer, "Trial Average")
        self.emin, self.emax = transformer.ci_low_, transformer.ci_high_
        self.grid = self.grid[: self.emin.shape[0]]
        return transformed

    def interp(self):
        transformer = transformers.PeriodicGridTransformer(
            period=self.period,
            sampling_in=self.tr,
            start_offset=self.offset,
        )
        transformed = self._transform(transformer, "Crop and Interpolate")
        self.grid = transformer.target_grid_ - self.offset
        return transformed

    def fft(self):
        transformer = transformers.FFTTransformer(self.sampling_rate)
        return self._transform(transformer, "FFT")

    def preprocess(self):
        self.reset()
        self = self.interp().psc().average().trial_average(bootstrap=True)
        return self.transformed_data.squeeze()

    @classmethod
    def from_nifti(
        cls, mask: str, data: str, period: float, labels=None, stimulus_offset=14
    ):
        masker = NiftiMasker(mask_img=mask)
        dat = nib.load(data)
        ts = masker.fit_transform(data)
        return cls(
            tr=dat.header["pixdim"][4],
            period=period,
            data=ts,
            labels=labels,
            stimulus_offset=stimulus_offset,
        )

    def plot(self, plotci: bool = True):
        fig, ax = plt.subplots(dpi=150)
        ax.plot(self.grid[: self.transformed_data.size], self.transformed_data)
        ax.set(ylabel="Amplitude", xlabel="Time")
        if self.emin is not None and plotci:
            ax.fill_between(
                self.grid[: self.transformed_data.size],
                self.emin.squeeze(),
                self.emax.squeeze(),
                alpha=0.4,
            )
