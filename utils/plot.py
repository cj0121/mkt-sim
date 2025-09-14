import matplotlib.pyplot as plt
from typing import Optional


def plt_plot(
    x,
    y,
    label: str = "",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize=(8, 5),
    *,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    return_ax: bool = False,
    **plot_kwargs,
):
    """Lightweight plotting helper.

    - Reuses an axes if provided (faster in notebooks) or creates a new figure/axes.
    - Avoids legend call when no label is provided.
    - Supports custom matplotlib line kwargs via **plot_kwargs.
    - Optionally returns the axes for further composition.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Defaults that users can override
    default_style = {"linestyle": "-", "linewidth": 2}
    default_style.update(plot_kwargs)
    ax.plot(x, y, label=label, **default_style)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(True, linestyle="--", alpha=0.6)
    if label:
        ax.legend()

    if show:
        fig.tight_layout()
        plt.show()

    if return_ax:
        return ax