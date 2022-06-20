"""
Test the transformers
"""
import numpy as np
from numpy.testing import assert_array_almost_equal
from fluctus.transformers import (
    PSCScaler,
    PeriodicGridTransformer,
    FFTTransformer,
)


def make_stimulus(nvols, offset, tr, frequency):
    """Create a sinusoidal that mimics the oscillations I use in my visual
    experiments.
    """
    y_offset, y_norm = 1, 2
    exponent, luminance = 1, 1
    x = np.array([x * tr for x in range(nvols)])

    period = 1 / frequency
    _, time = np.divmod(offset, period)
    phase_offset = (time / period) * (2 * np.pi) + np.pi
    osc = (
        (np.cos(2 * np.pi * frequency * x - phase_offset) + y_offset) / y_norm
    ) ** exponent * luminance
    osc = np.where(x < offset, 0, osc)
    return osc


def test_psc_transform():
    """
    Test that PSC normalization is working as expected.
    """
    in_data = np.array([[0.5, 1.5, 1], [1, 1.1, 0.9]]).T
    psc = PSCScaler()
    out = psc.fit_transform(in_data)
    expects = np.array([[-50, 50, 0], [0, 10, -10]]).T
    assert_array_almost_equal(expects, out)


def test_psc_inverse_transform():
    """
    Test that PSC inverse_transform is also working as expected.
    """
    in_data = np.array([[0.5, 1.5, 1], [1, 1.1, 0.9]]).T
    psc = PSCScaler()
    out = psc.fit_transform(in_data)
    inverse = psc.inverse_transform(out)
    assert_array_almost_equal(in_data, inverse)


def test_periodic_grid_transformer_grids():
    """
    Test that the periodic grid transformer is generating correct target grids.
    The performed interpolation is done by scipy, so we take that's already tested.
    """
    # We start with a sinusoisal signal with a 0.1s grid.
    nvols, offset, tr, frequency = 100, 0, 1, 0.2
    period = 1 / frequency
    stim = make_stimulus(nvols, offset, tr, frequency)
    transf = PeriodicGridTransformer(period)
    transf.fit(stim)
    # test that the grids are working.
    assert transf.source_grid_[-1] == nvols - 1
    # the target grid has to fit enough periods, and be off by one sample from
    # the start of the next period.
    assert (
        transf.target_grid_[-1]
        == transf.source_grid_[-1] // period * period - transf.sampling_out_
    )


def test_periodic_grid_transformer_transforms():
    """
    Test that the periodic grid transformer is working.
    This is just a sanity check that there is nothing wrong in running the
    current implementation.
    """
    # We start with a sinusoisal signal with a 0.1s grid.
    nvols, offset, tr, frequency = 100, 0, 1, 0.2
    period = 1 / frequency
    stim = make_stimulus(nvols, offset, tr, frequency)
    transf = PeriodicGridTransformer(period)
    transf.fit_transform(stim)


def test_fft_transformer_normalization():
    """
    Test that the returned power spectrum units make sense.
    """
    nvols, offset, tr, frequency = 100, 0, 1, 0.2
    stim = make_stimulus(nvols, offset, tr, frequency)
    fft = FFTTransformer(tr)
    px = fft.fit_transform(stim[:, np.newaxis])
    assert px[0] == stim.mean()
