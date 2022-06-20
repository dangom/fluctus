import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, input_data
from fluctus.preprocessing import (
    FeatureAverager,
    FFTTransformer,
    PeriodicGridTransformer,
    PSCScaler,
    TrialAveragingTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

period = 10
stimulus_offset = 14


os.chdir("/Users/dangom/MGH/data")

# Step 1. Define the localizer mask and constrain it to V1.
locf = "activation.nii.gz"
v1f = "testv1.nii.gz"
data = "bold.nii.gz"


def get_tr(data):
    ni = nib.load(data)
    return ni.header["pixdim"][4]


def get_offset(stimulus_offset=14, period=10):
    offset = stimulus_offset
    # 39.999 just so we accept 54 as valid. Python doesn't have a do while loop.
    while offset <= (stimulus_offset + 39.999):  # 40 seconds after start of stim
        offset += period
    return offset


def truncate_affine(nii):
    return image.new_img_like(
        nii, nii.get_fdata(), np.around(nii.affine, 5), copy_header=True
    )


def make_mask(v1, loc, threshold):
    mask = image.math_img("v1 * loc", v1=v1, loc=loc)
    return image.math_img(
        "f.astype(bool).astype(int)", f=image.binarize_img(mask, threshold=threshold)
    )


tr = get_tr(data)
offset = get_offset(stimulus_offset, period)

v1 = truncate_affine(image.load_img(v1f))
loc = truncate_affine(image.load_img(locf))

mask = make_mask(v1, loc, 10)

masker = input_data.NiftiMasker(mask_img=mask, standardize=False)
masker.fit(data)
# breakpoint()
report = masker.generate_report()
# report.save_as_html(snakemake.output["report"])


# Step 2 tag which localizer voxels are vessels by getting the top ~ 10%
# of voxels within mask
activation = masker.fit_transform(loc)
n_vessels = int(activation.size // 10)
ind = np.argpartition(activation.squeeze(), -n_vessels)

print(f"{activation.shape=}")
print(f"{n_vessels=}")

# Step 3 - pipelines for analysis.
pipe = Pipeline(
    [
        ("masker", masker),
        (
            "interp",
            PeriodicGridTransformer(period=period, sampling_in=tr, start_offset=offset),
        ),
        ("scale", PSCScaler()),
    ]
)

ts_psc = pipe.fit_transform(data)

n_trials = pipe["interp"].n_periods_
interp_sampling_rate = pipe["interp"].sampling_out_


min_max = Pipeline(
    [
        ("tatminmax", TrialAveragingTransformer(n_trials, bootstrap=False)),
        ("minmax", MinMaxScaler()),
    ]
)
min_max.fit(ts_psc)
amplitudes = min_max["minmax"].data_range_
labels = [
    "vein" if x in ind[-n_vessels:] else "parenchyma" for x in range(activation.size)
]
dfa = pd.DataFrame(zip(labels, amplitudes), columns=["ROI", "amplitude"])
dfa["frequency"] = 1 / period
dfa.to_csv()

col_trans = ColumnTransformer(
    [
        ("vein", FeatureAverager(), ind[-n_vessels:]),
        ("parenchyma", FeatureAverager(), ind[:-n_vessels]),
    ]
)

# Step 4. get power spectrum and averaged oscillation.
ts = col_trans.fit_transform(ts_psc)
print(f"{ts.shape=}")


print(f"{n_trials=}")
print(f"{interp_sampling_rate=}")

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

# px_df.to_csv(snakemake.output["px"], index=False)

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
# osc_df.to_csv(snakemake.output["osc"], index=False)


# 5.1 Plot timeseries AND bootstrap confidence
fig, (axt, axf) = plt.subplots(nrows=2, dpi=100, figsize=plt.figaspect(0.7))

axt.plot(targetgrid, ts[:, 0], label="vein")
axt.plot(targetgrid, ts[:, 1], label="parenchyma")
axt.set(ylabel="% BOLD", xlabel="Time [s]")
axt.legend(ncol=2, frameon=False, loc="lower center")

idx = fft.closest_freq_index(0.6)
axf.plot(freqs[:idx], px[:idx, 0], label="vein")
axf.plot(freqs[:idx], px[:idx, 1], label="parenchyma")
axf.set(ylabel="Power [% BOLD]", xlabel="Frequency [Hz]")
# axf.axvspan(1/period - 0.005,1/period + 0.005, alpha=0.1, color="red")
idx_stim = fft.closest_freq_index(1 / period)
axf.annotate(
    "Stimulus",
    (freqs[idx_stim] - 0.001, px[idx_stim, 0] + 0.02),
    size="small",
    xytext=(-50, -10),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)
# axf.axvline(1/period)

axin = axf.inset_axes([0.55, 0.55, 0.4, 0.4])
# axin.spines["right"].set_visible(False)
# axin.spines["top"].set_visible(False)
# axin.spines["left"].set_visible(False)
# axin.spines["bottom"].set_visible(False)


axin.plot(oscgrid, osc[:, 0], label="vein")
# axin.fill_between(oscgrid, emin[:, 0], emax[:, 0], alpha=0.2)
axin.plot(oscgrid, osc[:, 1], label="parenchyma")
# axin.fill_between(oscgrid, emin[:, 1], emax[:, 1], alpha=0.2)
axin.set(
    xlabel=f"Period = {period:.2f}s",
    xticklabels=[],
    yticklabels=[],
    xticks=[],
    yticks=[],
)

fig.suptitle(f"Response from {activation.size} voxels in V1")
