"""Тесты графиков Plotly дашборда: состав трасс, стиль и договор со страницей."""
import pandas as pd
import pytest

from src.dashboard import SCORED_TEST_PATH, key_numbers
from src.dashboard_figures import CONFIG, COBALT, SAFFRON, figure_specs
from src.overbooking import OVERBOOKING_PATH

FRAME = pd.read_csv(SCORED_TEST_PATH)
DATES = pd.read_csv(OVERBOOKING_PATH)


@pytest.fixture(name="specs", scope="module")
def fixture_specs():
    """Графики на срезе по умолчанию."""
    return figure_specs(FRAME, DATES)


def test_figure_traces(specs):
    """Пять графиков, у каждого — трассы нужных типов в оговорённом порядке."""
    kinds = {name: [trace["type"] for trace in spec["data"]] for name, spec in specs.items()}
    assert kinds == {"queue": ["bar", "bar"], "curves": ["scatter", "scatter", "scatter"],
                     "weeks": ["bar", "scatter", "scatter"], "heatmap": ["heatmap"], "segments": ["bar", "bar"]}


def test_figure_style(specs):
    """Шрифт Golos Text, десятичная запятая, цвета палитры; числа — обычными списками, а не bdata."""
    for spec in specs.values():
        assert spec["layout"]["font"]["family"].startswith("Golos Text")
        assert spec["layout"]["separators"] == ", "
    assert specs["queue"]["data"][0]["marker"]["color"] == SAFFRON
    assert specs["curves"]["data"][1]["line"]["color"] == COBALT
    assert isinstance(specs["queue"]["data"][0]["y"], list) and len(specs["queue"]["data"][0]["y"]) == 100
    assert isinstance(specs["curves"]["data"][1]["y"], list) and len(specs["curves"]["data"][1]["y"]) == 101


def test_figure_defaults(specs):
    """Черта и точка стоят на пороге по умолчанию (70 %), потери в точке — 0,56 млн €."""
    pct = key_numbers(FRAME)["threshold_pct"]
    assert specs["queue"]["layout"]["shapes"][0]["x0"] == pytest.approx(pct + 0.5)
    assert "предоплата с 70 %" in specs["queue"]["layout"]["annotations"][0]["text"]
    assert specs["curves"]["data"][2]["x"] == [pct]
    assert specs["curves"]["data"][2]["y"][0] == pytest.approx(0.56, abs=0.005)


def test_heatmap_and_segments(specs):
    """Тепловая карта и сегменты — русские названия; пустая клетка — None и прочерк."""
    heat = specs["heatmap"]["data"][0]
    assert "Онлайн-агентства" in heat["y"] and heat["x"] == ["май", "июн", "июл", "авг"]
    assert any(value is None for row in heat["z"] for value in row) == any(
        text == "–" for row in heat["text"] for text in row)
    assert specs["segments"]["data"][0]["orientation"] == "h"
    assert "Онлайн-агентства" in specs["segments"]["data"][0]["y"]
    layout = specs["segments"]["layout"]
    assert layout["legend"]["y"] == 1.02 and layout["legend"]["yanchor"] == "bottom" and layout["margin"]["t"] >= 36


def test_config_offline():
    """Без логотипа Plotly и без всего, что тянет файлы из сети."""
    assert CONFIG["displaylogo"] is False and CONFIG["locale"] == "ru"
    assert not {"mathjax", "topojsonURL"} & set(CONFIG)
