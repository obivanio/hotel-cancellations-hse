"""Тесты вспомогательных функций для графиков."""
import pandas as pd
from matplotlib.figure import Figure

from src.plots import bar_rates, save_figure


def test_save_figure(tmp_path):
    """График сохраняется в PNG с заданным именем."""
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    path = save_figure(fig, "test", tmp_path)
    assert path == tmp_path / "test.png"
    assert path.stat().st_size > 0


def test_bar_rates():
    """Один столбец на группу, заголовок проставлен."""
    rates = pd.DataFrame({"cancel_rate": [0.1, 0.4]}, index=["a", "b"])
    fig = Figure()
    ax = bar_rates(rates, fig.add_subplot(111), "заголовок")
    assert len(ax.patches) == 2
    assert ax.get_title() == "заголовок"
