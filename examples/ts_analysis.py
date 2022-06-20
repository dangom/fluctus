import re
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

files = glob("/Users/dangom/MGH/vaso/*/px/*osc.csv")

def subject_from_run_name(run_name: str) -> str:
    """
    From a functional dataset prefix (run_name) such as
    'task-0p16_run-02', extract the frequency.
    e.g.: 'task-0p16_run-02' -> 0.16
    """
    return run_name.split("/")[5]

def frequency_from_run_name(run_name: str) -> float:
    """
    From a functional dataset prefix (run_name) such as
    'task-0p16_run-02', extract the frequency.
    e.g.: 'task-0p16_run-02' -> 0.16
    """
    freq_name: str = re.search(r"task-0?p(\d+)_", run_name).group(1)
    if freq_name == "033":
        return 1 / 30
    elif freq_name == "0167":
        return 1 / 60
    else:
        return float("0." + freq_name)

def run_from_run_name(run_name: str) -> str:
    return int(re.search(r"task-0?p(\d+)_run-(\d\d)", run_name).group(2))

def prep_data(data: str):
    df = pd.read_csv(data, index_col=0)
    df["subject"] = subject_from_run_name(data)
    df["run"] = run_from_run_name(data)
    return df

def prep_data2(data: str):
    df = pd.read_csv(data, index_col=0)
    df["subject"] = subject_from_run_name(data)[:-1]
    df["frequency"]  = frequency_from_run_name(data)
    df["run"] = run_from_run_name(data)
    del df["ci_low"]
    del df["ci_high"]
    df = df.reset_index().rename(columns={"index": "voxel"})
    return df


df = pd.concat([prep_data2(f) for f in files])
df["frequency"] = df["frequency"].astype(float)

