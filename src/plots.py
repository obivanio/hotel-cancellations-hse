"""Единый стиль графиков и сохранение их в reports/figures."""
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.pipeline import ROOT

FIGURES_DIR = ROOT / "reports" / "figures"


def set_style() -> None:
    """Единый стиль графиков проекта."""
    sns.set_theme(style="whitegrid", context="notebook", rc={"figure.figsize": (9, 5)})


def save_figure(fig: Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Сохранить график в PNG и вернуть путь к файлу."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def bar_rates(rates: pd.DataFrame, ax: Axes, title: str, column: str = "cancel_rate") -> Axes:
    """Горизонтальные столбцы доли отмен по группам с подписями в процентах."""
    values = rates[column]
    ax.barh([str(label) for label in values.index], values.to_numpy())
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.bar_label(ax.containers[0], labels=[f"{value:.0%}" for value in values], padding=3)
    ax.set_title(title)
    ax.set_xlabel("доля отмен")
    return ax
