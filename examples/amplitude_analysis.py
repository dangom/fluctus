from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.api.types import CategoricalDtype
from scipy.stats import sem
from seaborn._statistics import KDE
from sklearn.linear_model import LinearRegression
from symfit import Fit, Model, exp, parameters, variables

files = glob("/Users/dangom/MGH/vaso/*/amp/*.csv")


def subject_from_run_name(run_name: str) -> str:
    """
    From a functional dataset prefix (run_name) such as
    'task-0p16_run-02', extract the frequency.
    e.g.: 'task-0p16_run-02' -> 0.16
    """
    return run_name.split("/")[5]


def run_from_run_name(run_name: str) -> str:
    """
    From a functional dataset prefix (run_name) such as
    'task-0p16_run-02', extract the frequency.
    e.g.: 'task-0p16_run-02' -> 0.16
    """
    return run_name.split("/")[7][-17:-15]


def prep_data(data: str):
    df = pd.read_csv(data, index_col=0)
    df["subject"] = subject_from_run_name(data)
    df["run"] = run_from_run_name(data)
    return df


def prep_data2(data: str):
    df = pd.read_csv(data, index_col=0)
    df["subject"] = subject_from_run_name(data)[:-1]
    df["run"] = run_from_run_name(data)
    df = df.reset_index().rename(columns={"index": "voxel"})
    return df


df = pd.concat([prep_data2(f) for f in files])
df["frequency"] = df["frequency"].astype(float)


pivotted = df.pivot_table(
    index=["subject", "voxel"],
    columns=["frequency", "ROI"],
    aggfunc=np.nanmean,
    fill_value=np.nan,
)

amps = pivotted["amplitude"]
ratio_20_05 = (amps[0.20] / amps[0.05]).unstack(0).melt().dropna()
ratio_20_05["ratio"] = "0.20/0.05Hz"
ratio_10_05 = (amps[0.10] / amps[0.05]).unstack(0).melt().dropna()
ratio_10_05["ratio"] = "0.10/0.05Hz"
ratio_20_10 = (amps[0.20] / amps[0.10]).unstack(0).melt().dropna()
ratio_20_10["ratio"] = "0.20/0.10Hz"
ratios = pd.concat([ratio_20_05, ratio_10_05])

ratio_cat = CategoricalDtype(categories=["0.10/0.05Hz", "0.20/0.05Hz"], ordered=True)
ratios["ratio"] = ratios["ratio"].astype(ratio_cat)
ratios.replace([np.inf, -np.inf], np.nan, inplace=True)

hrf_expect = {
    "parenchyma": {
        "10": 0.7701306688833854,
        "20": 0.25474149446080396,
        "10/20": 0.3307769768864735,
    },
    "vein": {
        "10": 0.7224597180920983,
        "20": 0.21382414759620316,
        "10/20": 0.29596687848684333,
    },
    "canonical": {
        "10": 0.524519027240252,
        "20": 0.06911398510232773,
        "10/20": 0.1317664021951116,
    },
}
ratios = ratios.rename(columns={"ROI": "compartment"})
g = sns.catplot(
    data=ratios,
    x="subject",
    hue="compartment",
    y="value",
    col="ratio",
    kind="bar",
    hue_order=["vein", "parenchyma"],
    col_order=["0.10/0.05Hz", "0.20/0.05Hz"],
    alpha=0.8,
    edgecolor="black",
)
ax1 = g.axes[0][0]
ax2 = g.axes[0][1]

ax1.axhline(hrf_expect["vein"]["10"], color="C0")
ax1.axhline(hrf_expect["parenchyma"]["10"], color="C1")

vpred = ax2.axhline(hrf_expect["vein"]["20"], color="C0", label="Vein HRF prediction")
ppred = ax2.axhline(
    hrf_expect["parenchyma"]["20"], color="C1", label="Parenchyma HRF prediction"
)

ax1.axhline(hrf_expect["canonical"]["10"], color="black")
cpred = ax2.axhline(
    hrf_expect["canonical"]["20"], color="black", label="Canonical HRF prediction"
)
ax2.legend(
    [vpred, ppred, cpred],
    [vpred.get_label(), ppred.get_label(), cpred.get_label()],
    loc="upper center",
    frameon=False,
)
ax1.set(ylabel="Amplitude Attenuation [ratio of response amplitudes]")
plt.ylim([0, 1])

plt.savefig("/Users/dangom/MGH/vaso/ratiosplot.png", dpi=300)

df["amplitude"] = df["amplitude"] / 2  # oscillation goes up and down.
g = sns.barplot(
    data=df,
    y="amplitude",
    x="frequency",
    hue="ROI",
    edgecolor="black",
    alpha=0.8,
    hue_order=["vein", "parenchyma"],
)
labels = g.get_xticklabels()  # get x labels
for i, l in enumerate(labels):
    if i % 2 == 0:
        labels[i] = ""  # skip even labels
    else:
        label = f"{float(l.get_text()):.3f}"
        labels[i].set_text(label)
g.set_xticklabels(labels, rotation=30)  # set new labels
# g.set_ylim([0, 5])
plt.savefig("/Users/dangom/MGH/vaso/allfreqs.png", dpi=300)
