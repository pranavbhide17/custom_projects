from itertools import cycle
from typing import List, Optional

from  matplotlib import pyplot
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse


def scatter_plot(
    transformed_data: pd.DataFrame,
    color: str = "y",
    size: float = 20.0,
    splot: Optional[pyplot.Axes] = None,
    label: Optional[List[str]] = None,
):
    """Draw a 2D scatter plot of transformed PCA data."""
    if splot is None:
        fig, splot = pyplot.subplots()
    columns = transformed_data.columns
    splot.scatter(
        transformed_data[columns[0]],
        transformed_data[columns[1]],
        s=size,
        c=color,
        label=label,
        alpha=0.6,
        edgecolors='w'
    )
    splot.set_aspect("equal", "box")
    splot.set_xlabel("Component 1")
    splot.set_ylabel("Component 2")
    if label:
        splot.legend()
    return splot


def plot_density_estimation_results(
    X: pd.DataFrame,
    Y_: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    title: str,
):
    """Use this function to plot the estimated distribution"""
    color_iter = cycle(["navy", "c", "cornflowerblue", "gold", "darkorange", "g"])
    pyplot.figure()
    splot = pyplot.subplot()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        scatter_plot(X.loc[Y_ == i], color=color, splot=splot)
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    pyplot.title(title)



def plot_finnish_parties(transformed_data: pd.DataFrame, splot: Optional[pyplot.Axes] = None):
    """Highlight Finnish parties in a 2D PCA plot by color group."""
    finnish_parties = [
        {"parties": ["SDP", "VAS", "VIHR"], "country": "fin", "color": "r"},
        {"parties": ["KESK", "KD"], "country": "fin", "color": "g"},
        {"parties": ["KOK", "SFP"], "country": "fin", "color": "b"},
        {"parties": ["PS"], "country": "fin", "color": "k"},
    ]

    if splot is None:
        fig, splot = pyplot.subplots()

    for group in finnish_parties:
        mask = (transformed_data.index.get_level_values("country") == group["country"]) & \
               (transformed_data.index.get_level_values("party").isin(group["parties"]))
        group_data = transformed_data[mask]
        scatter_plot(group_data, color=group["color"], splot=splot, label=group["parties"])

    splot.set_title("Finnish Parties (PCA Space)")
    #pyplot.show()