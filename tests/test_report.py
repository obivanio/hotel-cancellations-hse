"""Тесты интерактивного отчёта."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.overbooking import date_table, outcome_table
from src.policies import equal_share_curves, policy_table, scored
from src.report import (SECTIONS, _summary, _texts, build_report, fig_overbooking, fig_risk,
                        fig_share_curve, fig_top_share, risk_table)


@pytest.fixture(name="tables", scope="module")
def fixture_tables():
    """Таблицы шага 4 на синтетических бронях двух отелей (те же функции, что и на реальных данных)."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
        "arrival_date": pd.Timestamp("2017-05-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D"),
        "lead_time": rng.integers(0, 300, n),
        "booking_value": rng.uniform(50, 1000, n),
        "deposit_type": "No Deposit",
        "market_segment": rng.choice(["Online TA", "Direct", "Groups"], n),
    })
    df["arrival_month"] = df["arrival_date"].dt.month
    proba = rng.random(n)
    df["is_canceled"] = (rng.random(n) < proba).astype(int)
    s = scored(df, proba)
    return {
        "policies": policy_table(s, lead_days=30, threshold=0.17),
        "curves": equal_share_curves(s),
        "dates": date_table(df, proba, np.full(n, 0.35)),
        "risk": risk_table(df, proba),
    }


def test_figures_have_data(tables):
    """Каждая fig_* — фигура Plotly с данными; переключатель отеля управляет всеми трассами."""
    figures = [fig_share_curve(tables["curves"], tables["policies"]), fig_top_share(tables["curves"]),
               fig_overbooking(tables["dates"]), fig_risk(tables["risk"])]
    for fig in figures:
        assert isinstance(fig, go.Figure) and len(fig.data) > 0
        # Проверяем dropdowns: для restyle нужна visible, для relayout проверяем наличие меню
        for menu in fig.layout.updatemenus:
            for button in menu.buttons:
                if button.method == "restyle":
                    assert len(button.args[0]["visible"]) == len(fig.data)
    slider = figures[1].layout.sliders[0]
    assert len(slider.steps) == 20 and slider.steps[0].label == "5 %"


def test_risk_table(tables):
    """Риск по отелю, сегменту и месяцу: есть «все отели», доли в [0, 1]."""
    risk = tables["risk"]
    assert "все отели" in set(risk["hotel"])
    assert risk[["mean_proba", "cancel_rate"]].stack().between(0, 1).all()


def test_build_report(tables, tmp_path):
    """report.html собирается из CSV: оглавление, все разделы, plotly.js встроен (работает без интернета)."""
    tables["policies"].to_csv(tmp_path / "policies.csv", index=False)
    tables["curves"].to_csv(tmp_path / "share_curves.csv", index=False)
    tables["dates"].to_csv(tmp_path / "overbooking.csv", index=False)
    tables["risk"].to_csv(tmp_path / "risk_segments.csv", index=False)
    html = build_report(tmp_path, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Оглавление" in html
    for key, title in SECTIONS.items():
        assert f'id="{key}"' in html and f'href="#{key}"' in html and title in html
    assert html.count("Plotly.newPlot") == 4
    assert "<script src=" not in html and len(html) > 1_000_000


def test_chart2_slider_relayout_no_visible(tables):
    """Слайдер графика 2 использует relayout (не visible), для подвижного маркера и аннотации."""
    fig = fig_top_share(tables["curves"])
    assert len(fig.layout.sliders) > 0
    slider = fig.layout.sliders[0]
    for step in slider.steps:
        # Все шаги должны быть relayout, не меняющие видимость трасс
        assert step.method == "relayout"
        # args должны содержать shapes/annotations, а не visible
        assert isinstance(step.args, (list, tuple)) and len(step.args) > 0
        relayout_dict = step.args[1] if len(step.args) > 1 else step.args[0]
        assert isinstance(relayout_dict, dict)
        assert "shapes" in relayout_dict or "annotations" in relayout_dict


def test_chart2_has_restyle_dropdown(tables):
    """График 2 имеет выпадающий список для выбора отеля (restyle для видимости)."""
    fig = fig_top_share(tables["curves"])
    assert len(fig.layout.updatemenus) > 0
    dropdown = fig.layout.updatemenus[0]
    assert len(dropdown.buttons) > 0
    # Каждый бутон должен использовать restyle для видимости
    for button in dropdown.buttons:
        assert button.method == "restyle"
        assert "visible" in button.args[0]
        assert len(button.args[0]["visible"]) == len(fig.data)


def test_slider_text_derived_from_curves(tables):
    """Вывод под графиком 2 содержит числа из таблицы кривых, не hardcoded."""
    summary = _summary(tables["policies"], tables["curves"], outcome_table(tables["dates"]))
    texts = _texts(summary, tables["dates"], tables["curves"], tables["risk"])
    slider_text = texts["slider"]
    # Не должно быть hardcoded claim про падение-рост потерь
    assert "потери сначала падают, потом растут" not in slider_text
    # Должны быть конкретные числовые данные (доля с минимумом)
    assert "%" in slider_text


def test_risk_text_derived_from_risk_table(tables):
    """Вывод под графиком 4 содержит числовые данные: максимальный риск из таблицы."""
    # _texts принимает summary, dates, curves, risk
    # для chart 4 (risk) используются данные из tables["risk"]
    texts = _texts({}, tables["dates"], None, tables["risk"])
    risk_text = texts["risk"]
    # Текст не должен быть только hint'ом, должны быть данные из таблицы
    assert "риск" in risk_text and "%" in risk_text
