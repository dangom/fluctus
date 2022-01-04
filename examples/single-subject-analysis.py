from osclib.transformers import (
    PeriodicGridTransformer,
    PSCScaler,
    FFTTransformer,
    FeatureAverager,
    TrialAveragingTransformer,
)
from nilearn import image, input_data
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import os
import pandas as pd
import nibabel as nib

period = 10
stimulus_offset = 14


def get_tr(data):
    ni = nib.load(data)
    return ni.header["pixdim"][4]


def get_offset(stimulus_offset=14, period=10):
    offset = stimulus_offset
    # 39.999 just so we accept 54 as valid. Python doesn't have a do while loop.
    while offset <= (stimulus_offset + 39.999):  # 40 seconds after start of stim
        offset += period
    return offset


os.chdir("/Users/dangom/MGH/data")

# Step 1. Define the localizer mask and constrain it to V1.
loc = "activation.nii.gz"
v1 = "testv1.nii.gz"
data = "bold.nii.gz"

tr = get_tr(data)
offset = get_offset(stimulus_offset, period)


mask = image.binarize_img(
    image.math_img(
        "v1 * loc",
        v1=image.load_img(v1),
        loc=image.load_img(loc),
    ),
    10,
)

masker = input_data.NiftiMasker(mask_img=mask, standardize=False)

# Step 2 tag which localizer voxels are vessels by getting the top ~ 10%
# of voxels within mask
activation = masker.fit_transform(loc)
n_vessels = int(activation.size // 10)
ind = np.argpartition(activation.squeeze(), -n_vessels)

# Step 3 - pipelines for analysis.
pipe = Pipeline(
    [
        ("masker", masker),
        (
            "interp",
            PeriodicGridTransformer(period=period, sampling_in=tr, start_offset=offset),
        ),
        ("scale", PSCScaler()),
        (
            "roisel",
            ColumnTransformer(
                [
                    ("vein", FeatureAverager(), ind[-n_vessels:]),
                    ("parenchyma", FeatureAverager(), ind[:-n_vessels]),
                ]
            ),
        ),
    ]
)

# Step 4. get power spectrum and averaged oscillation.
ts = pipe.fit_transform(data)

n_trials = pipe["interp"].n_periods_
interp_sampling_rate = pipe["interp"].sampling_out_
fft = FFTTransformer(interp_sampling_rate)
tat = TrialAveragingTransformer(n_trials)

px = fft.fit_transform(ts)
osc = tat.fit_transform(ts)

# Step 5. Make plots and save the data
emin, emax = tat.ci_low_, tat.ci_high_
# Sourcegrid is the grid of timestamps in ms from the original data.
# Targetgrid is the grid of timestamps in ms from the interpolated data.
sourcegrid, targetgrid = pipe["interp"].source_grid_, pipe["interp"].target_grid_
# Oscgrid is the grid of timestamps of the trial averaged data.
oscgrid = (targetgrid - offset)[: osc.shape[0]]
freqs = fft.freqs_

# Save all of the data to csv
px_df = (
    pd.DataFrame(px, columns=["vessel", "parenchyma"], index=freqs)
    .reset_index()
    .melt(
        value_vars=["vessel", "parenchyma"],
        id_vars=["index"],
        var_name="ROI",
        value_name="power",
    )
    .rename({"index": "frequency"}, axis=1)
)

# px_df.to_filename()

# Save the timeseries
osc_df = (
    pd.DataFrame(osc, columns=["vessel", "parenchyma"], index=oscgrid)
    .reset_index()
    .melt(
        value_vars=["vessel", "parenchyma"],
        id_vars=["index"],
        var_name="ROI",
        value_name="amplitude",
    )
    .rename({"index": "time"}, axis=1)
)
osc_df["ci_low"] = pd.Series(emin.T.flatten())
osc_df["ci_high"] = pd.Series(emax.T.flatten())
# osc_df.to_filename()

# 5.1 Plot timeseries AND bootstrap confidence
fig, (axt, axf) = plt.subplots(nrows=2, dpi=100, figsize=plt.figaspect(0.7))
axt.plot(oscgrid, osc[:, 0], label="vein")
axt.fill_between(oscgrid, emin[:, 0], emax[:, 0], alpha=0.2)
axt.plot(oscgrid, osc[:, 1], label="parenchyma")
axt.fill_between(oscgrid, emin[:, 1], emax[:, 1], alpha=0.2)
axt.set(ylabel="Amplitude [% BOLD]", xlabel="Time [s]")
axt.legend()

idx = fft.closest_freq_index(0.25)
axf.plot(freqs[:idx], px[:idx, 0], label="vein")
axf.plot(freqs[:idx], px[:idx, 1], label="parenchyma")
axf.set(ylabel="Power Spectrum [% BOLD]", xlabel="Frequency [Hz]")
axt.legend()
