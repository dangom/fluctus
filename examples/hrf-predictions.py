"""
This is essentially the code I used to generate the plots in our paper.
"""
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fluctus.hrf import HRF
from fluctus.interfaces import Oscillation
from fluctus.stimuli import SinusStim

SAVE_FIGURES = True
###############################################################################
## Create the HRFs

# Parameters from HRFs from Siero et al. (2011)
vein_hrf_params = {
    "peak_delay": 4.3,
    "peak_width": 0.59,
    "undershoot_delay": 7.5,
    "undershoot_width": 2.0,
    "positive_negative_ratio": 0.1,
    "amplitude": 4.5,
}

parenchyma_hrf_params = {
    "peak_delay": 3.9,
    "peak_width": 0.55,
    "undershoot_delay": 7.5,
    "undershoot_width": 2.0,
    "positive_negative_ratio": 0.1,
    "amplitude": 2.2,
}

t = np.arange(0, 16, 0.01)
vein_hrf = HRF(**vein_hrf_params)
par_hrf = HRF(**parenchyma_hrf_params)
can_hrf = HRF(amplitude=5)  # canonical Glover HRF

# Get an impulse response for the HRFs
vein_ir = vein_hrf.IR
par_ir = par_hrf.IR
can_ir = can_hrf.IR


###############################################################################
## Generate the predictions

tr = 0.01
tprime = np.arange(0, 5000, tr)

frequencies = np.arange(0.01, 0.3, 0.005)

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
        bbox=dict(facecolor="white", edgecolor="None", alpha=0.5, boxstyle="round,pad=0"),
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


def annotate(ax, what, x, y, color, style):
    text = ax.annotate(
        what,
        xy=(x, y),
        xycoords="data",
        xytext=(30, 5),
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

def get_predictions_for_frequencies_of_interest(what, compartment):

    amp = what

    # Predictions for frequencies of interest
    pred_of_interest = predictions[
        (np.isclose(predictions["frequency"], 0.05))
        | (np.isclose(predictions["frequency"], 0.10))
        | (np.isclose(predictions["frequency"], 0.20))
    ]

    def pred_of_interest_for(compartment: str):
        return pred_of_interest[pred_of_interest["model"] == compartment][amp]

    res = pred_of_interest_for(compartment)
    res.index = [0.05, 0.10, 0.20]
    return res


def plot_hrf_predictions(what, ylabel, formatter=lambda x: f"{x:.2f}", ax=None):
    """
    A little wrapper to generate the same plot for different measures, like amplitude, normalized_amplitude or delays.
    The variable names may not make much sense, since this plot was originally for amplitude (amp).
    """
    ## Plot 2.1 -- amplitudes
    if ax is None:
        fig, ax = plt.subplots(dpi=200)

    amp = what

    def amp_prediction_for(compartment: str):
        return predictions[predictions["model"] == compartment][amp]

    # Predictions for frequencies of interest
    freqs_of_interest = [0.05, 0.10, 0.20]
    pred_of_interest = predictions[
        (np.isclose(predictions["frequency"], 0.10))
        | (np.isclose(predictions["frequency"], 0.05))
        | (np.isclose(predictions["frequency"], 0.20))
    ]

    def pred_of_interest_for(compartment: str):
        return pred_of_interest[pred_of_interest["model"] == compartment][amp]

    (vplot,) = ax.plot(frequencies, amp_prediction_for("vein"), label="vein")
    (pplot,) = ax.plot(
        frequencies, amp_prediction_for("parenchyma"), label="parenchyma"
    )
    (cplot,) = ax.plot(frequencies, amp_prediction_for("canonical"), label="canonical")

    for plot, compartment in zip(
        [vplot, pplot, cplot], ["vein", "parenchyma", "canonical"]
    ):
        place_dots_on_curve(
            ax, plot, freqs_of_interest, pred_of_interest_for(compartment)
        )
        continue

        for i, f in enumerate(freqs_of_interest):
            txt = formatter(pred_of_interest_for(compartment).iloc[i])
            annotate(
                ax,
                txt,
                f,
                pred_of_interest_for(compartment).iloc[i],
                plot.get_color(),
                "arc3,rad=0.3",
            )

    # Add labels directly on the curves
    label_location = (20, 20)
    # place_label_on_curve(ax, "vein", vplot, *label_location, rotation=-72.5)
    # place_label_on_curve(ax, "parenchyma", pplot, *label_location, rotation=-53.0)
    # place_label_on_curve(ax, "canonical", cplot, *label_location, rotation=-63.0)

    ax.axvline(0.05, alpha=0.2, linestyle="dashed", zorder=-1, color="gray")
    ax.axvline(0.10, alpha=0.2, linestyle="dashed", zorder=-1, color="gray")
    ax.axvline(0.20, alpha=0.2, linestyle="dashed", zorder=-1, color="gray")

    ax.set(xlabel="Frequency [Hz]", ylabel=ylabel)
    # ax.legend(frameon=False)
    if ax is None:
        return fig, ax
    return ax


# Now plot the normalized_amplitude, the amplitude and the delays
plot_hrf_predictions("norm_amplitude", "Amplitude normalized to 0.05 Hz")
plot_hrf_predictions(
    "amplitude", "Response Amplitude (% Signal Change)", formatter=lambda x: f"{x:.2f}%"
)
plot_hrf_predictions("delay", "Response Delay (s)", formatter=lambda x: f"{x:.2f}s")




###############################################################################

# Figure 1. Figure has 6 plots, 2 rows and 3 columns.
# The first row is the HRF plot, with columns:
# 1. HRF plot 2. HRF response 3. What is amplitude and delay
# The second row is the HRF predictions, with columns:
# 1. amplitude 2. normalized amplitude 3. delay
sns.set_context("paper", font_scale=0.8)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
    dpi=100, nrows=2, ncols=3, figsize=(6.7, 3.8)
)

fig.suptitle("HRF models and  predictions")

ax1.set_title(r"$\bf{A.}$ HRF models", y=0.95)
(vplot,) = ax1.plot(t, vein_ir, label="vein")
(pplot,) = ax1.plot(t, par_ir, label="parenchyma")
(cplot,) = ax1.plot(t, can_ir, label="canonical")

place_label_on_curve(ax1, "vein", vplot, *(600, 600), rotation=-72.5)
place_label_on_curve(ax1, "parenchyma", pplot, *(520, 520), rotation=-59.0)
place_label_on_curve(ax1, "canonical", cplot, *(800, 800), rotation=-65.0)

# remove upper and right axes
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.set(xlabel="Time [s]", ylabel="% Signal Change")

stim = SinusStim(frequency=0.2).sample(tprime)
ax2.set_title(r"$\bf{B.}$ Responses (example 0.2Hz)", y=0.95)
ax2.plot(tprime[0:4000], vein_hrf.transform(stim, tr)[0:4000], label="vein")
ax2.plot(tprime[0:4000], par_hrf.transform(stim, tr)[0:4000], label="parenchyma")
ax2.plot(tprime[0:4000], can_hrf.transform(stim, tr)[0:4000], label="canonical")
(splot,) = ax2.plot(tprime[0:4000], stim[0:4000], alpha=0.5, color="gray", linewidth=0.5, linestyle="dashed")
place_label_on_curve(ax2, "Stimulus", splot, *(1700, 1700), rotation=0.)
ax2.set(ylim=[0, 15], xlabel="Time [s]", ylabel="% Signal Change")
ax2.legend(frameon=False, loc="upper center", ncol=3, fontsize=4)

# remove upper and right axes
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax3.set_title(r"$\bf{C.}$ Amplitude and Delay", y=0.95)
ax3.set(ylim=[-.2, 1.2], xlabel="Time [s]")
# remove y ticks
ax3.yaxis.set_ticks([])
ax3.xaxis.set_ticks([])
wave = SinusStim(frequency=0.1).sample(tprime)[400:1400]
t_wave = tprime[400:1400]
(wplot,) = ax3.plot(t_wave, wave)
(splot,) = ax3.plot(t_wave, SinusStim(frequency=0.1, start_offset=4).sample(t_wave), alpha=0.5, color="gray", linewidth=0.5, linestyle="dashed")
place_label_on_curve(ax3, "Response", wplot, *(400, 400), rotation=-60.)
place_label_on_curve(ax3, "Stimulus", splot, *(700, 700), rotation=-60.)
ax3.axhline(0.5, alpha=0.2, linestyle="dashed", zorder=-1, color="gray")
# Now plot an arrow from the peak to the mean
ax3.arrow(
    x=t_wave[np.argmax(wave)],
    y=0.,
    dx=0,
    dy=1.,
    length_includes_head=True,
    head_width=0.1,
    head_length=0.1,
    alpha=0.8,
    color="C0",
)

# add a text next to the arrow
ax3.text(
    t_wave[np.argmax(wave)] + 0.1,
    0.6,
    f"Amplitude",
    ha="left",
    va="center",
    fontsize=6,
    color="C0",
)
# despine
ax3.spines["left"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)

# Same arrow but now in x instead of y until the drought
ax3.arrow(
    0,
    0,
    t_wave[np.argmin(wave)],
    0,
    length_includes_head=True,
    head_width=0.1,
    head_length=0.1,
    alpha=0.8,
    color="C0",
)

# and the text
ax3.text(
    t_wave[np.argmin(wave)] - 0.5,
    0,
    f"Delay",
    ha="right",
    va="center",
    fontsize=6,
    color="C0",
)

ax3.set(xlim=[t_wave.min(), t_wave.max()])
plot_hrf_predictions(
    "amplitude",
    "% Signal Change",
    formatter=lambda x: f"{x:.2f}%",
    ax=ax4,
)
plot_hrf_predictions("norm_amplitude", "", ax=ax5)
plot_hrf_predictions(
    "delay", "Delay (s)", formatter=lambda x: f"{x:.2f}s", ax=ax6
)


ax4.set_title(r"$\bf{D.}$ Predicted Amplitude")
ax5.set_title(r"$\bf{E.}$ Predicted Normalized amplitude")
ax6.set_title(r"$\bf{F.}$ Predicted Delay")

# remove upper and right axes
ax4.spines["right"].set_visible(False)
ax4.spines["top"].set_visible(False)
# remove upper and right axes
ax5.spines["right"].set_visible(False)
ax5.spines["top"].set_visible(False)
# remove upper and right axes
ax6.spines["right"].set_visible(False)
ax6.spines["top"].set_visible(False)


# Tighten the layout
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig("/Users/dangom/MGH/osc/figures/1_hrf_predictions.png", dpi=300)

# TODO: add stimulus to 2nd plot. DONE
# TODO: add frequency that is used in second plot example DONE
# TODO: add letters to 6 panels DONE


#### Generate table with values for the three HRFs for the frequencies of interest.
dp = get_predictions_for_frequencies_of_interest("delay", "parenchyma")
dv = get_predictions_for_frequencies_of_interest("delay", "vein")
dc = get_predictions_for_frequencies_of_interest("delay", "canonical")
nap = get_predictions_for_frequencies_of_interest("norm_amplitude", "parenchyma")
nav = get_predictions_for_frequencies_of_interest("norm_amplitude", "vein")
nac = get_predictions_for_frequencies_of_interest("norm_amplitude", "canonical")
ap = get_predictions_for_frequencies_of_interest("amplitude", "parenchyma")
av = get_predictions_for_frequencies_of_interest("amplitude", "vein")
ac = get_predictions_for_frequencies_of_interest("amplitude", "canonical")

expectations = pd.DataFrame(columns=["frequency", "Model", "delay", "amplitude", "norm. amplitude"])
expectations.loc[0] = ["0.05Hz", "parenchyma", dp.iloc[0], ap.iloc[0], nap.iloc[0]]
expectations.loc[1] = ["0.05Hz", "vein", dv.iloc[0], av.iloc[0], nav.iloc[0]]
expectations.loc[2] = ["0.05Hz", "canonical", dc.iloc[0], ac.iloc[0], nac.iloc[0]]
expectations.loc[3] = ["0.10Hz", "parenchyma", dp.iloc[1], ap.iloc[1], nap.iloc[1]]
expectations.loc[4] = ["0.10Hz", "vein", dv.iloc[1], av.iloc[1], nav.iloc[1]]
expectations.loc[5] = ["0.10Hz", "canonical", dc.iloc[1], ac.iloc[1], nac.iloc[1]]
expectations.loc[6] = ["0.20Hz", "parenchyma", dp.iloc[2], ap.iloc[2], nap.iloc[2]]
expectations.loc[7] = ["0.20Hz", "vein", dv.iloc[2], av.iloc[2], nav.iloc[2]]
expectations.loc[8] = ["0.20Hz", "canonical", dc.iloc[2], ac.iloc[2], nac.iloc[2]]

expectations.to_csv("/Users/dangom/MGH/osc/expectations.csv")
