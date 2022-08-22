from dataclasses import dataclass

import numpy as np
from scipy.stats import gamma


@dataclass
class HRF:
    """Default values will return Glover's HRF for the auditory cortex."""

    peak_delay: float = 6.0
    peak_width: float = 0.9
    undershoot_delay: float = 12.0
    undershoot_width: float = 0.9
    positive_negative_ratio: float = 0.35
    first_gamma_only: bool = False
    second_gamma_only: bool = False
    amplitude: float = 1

    def sample(self, t: np.ndarray) -> np.ndarray:
        peak = gamma.pdf(
            t, self.peak_delay / self.peak_width, loc=0, scale=self.peak_width
        )
        undershoot = gamma.pdf(
            t,
            self.undershoot_delay / self.undershoot_width,
            loc=0,
            scale=self.undershoot_width,
        )
        peak_norm = peak.max()

        if self.first_gamma_only:
            return peak / peak_norm

        undershoot_norm = undershoot.max()

        if self.second_gamma_only:
            return -undershoot / undershoot_norm * self.positive_negative_ratio

        hrf = (
            peak / peak_norm
            - undershoot / undershoot_norm * self.positive_negative_ratio
        )
        return hrf * self.amplitude

    def transform(self, timeseries: np.ndarray, tr: float =0.02) -> np.ndarray:
        sample_times = np.arange(0, 32, tr)  # Sample 30 seconds at 20ms intervals.
        convolved = np.convolve(timeseries, self.sample(sample_times))
        # The return size of the convolved signal is len(signal) + len(sample_times) +1.
        # To get what we want we need to discard the extra time.
        return convolved[: -(len(sample_times) - 1)] * tr

    @property
    def IR(self) -> np.ndarray:
        "Impulse Response"
        t = np.arange(0, 16, 0.01)
        return self.sample(t)

    @property
    def fwhm(self) -> float:
        t = np.arange(0, 25, 0.001)
        hrf = self.sample(t)
        whr = np.where(np.abs(hrf - hrf.max() / 2) < 0.1)[0]
        return t[whr[-1]] - t[whr[0]]

    @property
    def ttp(self) -> float:
        t = np.arange(0, 25, 0.001)
        hrf = self.sample(t)
        return t[np.argmax(hrf)]

    def __repr__(self):
        return f"HRF(TTP={self.ttp:.2f}, FWHM={self.fwhm:.2f})"
