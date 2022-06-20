from dataclasses import dataclass
import numpy as np

@dataclass
class SinusStim:
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
