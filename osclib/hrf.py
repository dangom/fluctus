from dataclasses import dataclass

import numpy as np
from scipy.stats import gamma


@dataclass
class DoubleGammaHRF:
    """Default values will return Glover's HRF for the auditory cortex."""

    peak_delay: float = 6.0
    peak_width: float = 0.9
    undershoot_delay: float = 12.0
    undershoot_width: float = 0.9
    positive_negative_ratio: float = 0.35
    first_gamma_only: bool = False
    second_gamma_only: bool = False
    amplitude: float = 1

    def sample(self, t: np.array) -> np.array:
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

    def transform(self, timeseries: np.array, tr: float =0.02) -> np.array:
        sample_times = np.arange(0, 32, tr)  # Sample 30 seconds at 20ms intervals.
        convolved = np.convolve(timeseries, self.sample(sample_times))
        # The return size of the convolved signal is len(signal) + len(sample_times) +1.
        # To get what we want we need to discard the extra time.
        return convolved[: -(len(sample_times) - 1)] * tr

    @property
    def IR(self) -> np.array:
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


@dataclass
class SinusoidalStimulus:
    start_offset: float = 0
    frequency: float = 0.2
    exponent: float = 1.0
    luminance: float = 1.0
    extra_phase: float = 0.0

    def sample(self, t: np.array) -> np.array:
        period = 1 / self.frequency
        _, time = np.divmod(self.start_offset, period)
        phase_offset = (time / period) * (2 * np.pi) + np.pi
        y_offset, y_norm = 1, 2
        osc = (
            (
                np.cos(2 * np.pi * self.frequency * t - phase_offset + self.extra_phase)
                + y_offset
            )
            / y_norm
        ) ** self.exponent * self.luminance
        osc = np.where(t < self.start_offset, 0, osc)
        return osc
