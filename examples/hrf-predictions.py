import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fluctus.hrf import HRF
from fluctus.interfaces import Oscillation
from fluctus.stimuli import SinusStim


###############################################################################
## Create the HRFs

# Parameters from HRFs from Siero et al. (2011)
vein_hrf_params = {
    "peak_delay": 4.5,
    "peak_width": 0.53,
    "undershoot_delay": 7.5,
    "undershoot_width": 2.0,
    "positive_negative_ratio": 0.1,
    "amplitude": 4.5,
}

parenchyma_hrf_params = {
    "peak_delay": 4.1,
    "peak_width": 0.49,
    "undershoot_delay": 7.5,
    "undershoot_width": 2.0,
    "positive_negative_ratio": 0.1,
    "amplitude": 2.2,
}

t = np.arange(0, 16, 0.01)
vein_hrf = HRF(**vein_hrf_params)
par_hrf = HRF(**parenchyma_hrf_params)
can_hrf = HRF(amplitude=3.5)  # canonical Glover HRF

# Get an impulse response for the HRFs
vein_ir = vein_hrf.IR
par_ir = par_hrf.IR
can_ir = can_hrf.IR


###############################################################################
## Generate the predictions

tr = 0.01
tprime = np.arange(0, 5000, tr)

frequencies = np.arange(0.01, 0.4, 0.01)

# This idx is the index from which we consider to be in a steady-state
idx = np.abs((tprime - 2000.0)).argmin()


def response_amplitude(signal: np.array, start_idx: int):
    "Get max amplitude from oscillation as peak - drought amplitude"
    cropped_signal = signal[start_idx:]
    return (cropped_signal.max() - cropped_signal.min()) / 2


def response_delay(signal: np.array, tr: float, frequency: float):
    "Get max amplitude from oscillation as peak - drought amplitude"
    osc = Oscillation(tr, 1 / frequency, signal.reshape(-1, 1), 0)
    return osc.interp(0.01).psc().trial_average().phase[0]


# Loop through frequencies and collect the response amplitude and delay for each of the 3 hrfs
predictions = pd.DataFrame(
    columns=["model", "frequency", "amplitude", "norm_amplitude", "delay"]
)


reference_amplitudes_at_0p5 = {
    "vein": response_amplitude(
        vein_hrf.transform(SinusStim(frequency=0.05).sample(tprime), tr), idx
    ),
    "parenchyma": response_amplitude(
        par_hrf.transform(SinusStim(frequency=0.05).sample(tprime), tr), idx
    ),
    "canonical": response_amplitude(
        can_hrf.transform(SinusStim(frequency=0.05).sample(tprime), tr), idx
    ),
}

for i, freq in enumerate(frequencies):
    stim = SinusStim(frequency=freq).sample(tprime)
    for model, hrf in zip(
        ["vein", "parenchyma", "canonical"], [vein_hrf, par_hrf, can_hrf]
    ):
        response = hrf.transform(stim, tr)
        amp = response_amplitude(response, idx)
        delay = response_delay(response, tr, freq)
        norm_amp = amp / reference_amplitudes_at_0p5[model]
        predictions.loc[len(predictions.index)] = [model, freq, amp, norm_amp, delay]


###### Plotting ########################################################
## Helper function for plotting
def place_label_on_curve(ax, label, plot, x, y, rotation=0):
    ax.text(
        plot.get_xdata()[x],
        plot.get_ydata()[y],
        label,
        family="Roboto Condensed",
        bbox=dict(facecolor="white", edgecolor="None", alpha=0.5),
        color=plot.get_color(),
        ha="center",
        va="center",
        rotation=rotation,
    )


def place_dots_on_curve(ax, plot, x, y):
    ax.scatter(
        x,
        y,
        edgecolor=plot.get_color(),
        facecolor="white",
        zorder=20,
        s=22,
    )


def annotate(what, x, y, color, style):
    text = ax.annotate(
        what,
        xy=(x, y),
        xycoords="data",
        xytext=(20, 20),
        color=color,
        textcoords="offset points",
        ha="center",
        size="small",
        arrowprops=dict(
            arrowstyle="->",
            shrinkA=0,
            shrinkB=5,
            color=color,
            linewidth=0.75,
            connectionstyle=style,
        ),
    )
    text.set_path_effects(
        [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
    )
    text.arrow_patch.set_path_effects(
        [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
    )
    return text


###############################################################################
# Plot 1 - plot the HRFs

fig, ax = plt.subplots(dpi=200, figsize=plt.figaspect(0.5))

# Print the HRF info as a label
vlabel = f"TTP = {vein_hrf.ttp:.1f}s; FWHM={vein_hrf.fwhm:.1f}s"
plabel = f"TTP = {par_hrf.ttp:.1f}s; FWHM={par_hrf.fwhm:.1f}s"
clabel = f"TTP = {can_hrf.ttp:.1f}s; FWHM={can_hrf.fwhm:.1f}s"

(vplot,) = ax.plot(t, vein_ir, label=vlabel, clip_on=False)
(pplot,) = ax.plot(t, par_ir, label=plabel, clip_on=False)
(cplot,) = ax.plot(t, can_ir, label=clabel, clip_on=False)

# Despine for aesthetics
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_position(("data", -0.25))
# ax.spines["bottom"].set_position(("data", -0.3))

# Add labels directly on the curves
place_label_on_curve(ax, "vein", vplot, 500, 500, rotation=-72.5)
place_label_on_curve(ax, "parenchyma", pplot, 500, 500, rotation=-53.0)
place_label_on_curve(ax, "canonical", cplot, 700, 700, rotation=-63.0)

ax.legend(frameon=False)
ax.set(
    xlabel="Time [s]",
    ylabel="Percent Signal Change",
    title="Vein vs Parenchyma model HRFs",
)

###############################################################################
# Plot 2 - plot the HRF predictions


fig, ax = plt.subplots(dpi=200)


def amp_prediction_for(compartment: str):
    return predictions[predictions["model"] == compartment]["norm_amplitude"]


# Predictions for frequencies of interest
freqs_of_interest = [0.05, 0.10, 0.20]
pred_of_interest = predictions[
    (np.isclose(predictions["frequency"], 0.10))
    | (np.isclose(predictions["frequency"], 0.05))
    | (np.isclose(predictions["frequency"], 0.20))
]


def pred_of_interest_for(compartment: str):
    return pred_of_interest[pred_of_interest["model"] == compartment]["norm_amplitude"]


(vplot,) = ax.plot(frequencies, amp_prediction_for("vein"), label="vein")
(pplot,) = ax.plot(frequencies, amp_prediction_for("parenchyma"), label="parenchyma")
(cplot,) = ax.plot(frequencies, amp_prediction_for("canonical"), label="canonical")


for plot, compartment in zip([vplot, pplot, cplot], ["vein", "parenchyma", "canonical"]):
    place_dots_on_curve(ax, plot, freqs_of_interest, pred_of_interest_for(compartment))
    for i, f in enumerate((0.10, 0.20)):
        annotate(
            f"{pred_of_interest_for(compartment).iloc[i+1]:.2f}",
            f,
            pred_of_interest_for(compartment).iloc[i+1],
            plot.get_color(),
            "arc3,rad=0.3",
        )

ax.axvline(0.05, alpha=0.2, linestyle="dashed", zorder=-1, color="gray")
ax.axvline(0.10, alpha=0.2, linestyle="dashed", zorder=-1, color="gray")
ax.axvline(0.20, alpha=0.2, linestyle="dashed", zorder=-1, color="gray")


ax.set(xlabel="Frequency [Hz]", ylabel="Amplitude normalized to 0.05 Hz")
ax.legend(frameon=False)


###############################################################################
