"""Тесты интерактивного отчёта."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.overbooking import WALK_COSTS, date_table, outcome_table
from src.policies import equal_share_curves, policy_table, scored
from src.report import (MONTH_NAMES, SECTIONS, _assumptions_html, _risk_text, _slider_text, _summary, _texts,
                        build_report, fig_overbooking, fig_risk, fig_share_curve, fig_top_share, risk_table)


@pytest.fixture(name="tables", scope="module")
def fixture_tables():
    """Таблицы шага 4 на синтетических бронях двух отелей (те же функции, что и на реальных данных)."""
    rng = np.random.default_rng(0)
    n = 1200
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
    """Слайдер графика 2 только перерисовывает маркер (relayout), видимость трасс не трогает."""
    fig = fig_top_share(tables["curves"])
    for step in fig.layout.sliders[0].steps:
        assert step.method == "relayout"
        assert set(step.args[0]) == {"shapes", "annotations"}


def test_chart2_annotation_has_every_hotel(tables):
    """Подпись шага слайдера содержит числа по каждому отелю; маркер первого шага нарисован сразу."""
    fig = fig_top_share(tables["curves"])
    step = fig.layout.sliders[0].steps[3]
    text = step.args[0]["annotations"][0]["text"]
    model = tables["curves"][tables["curves"]["method"] == "модель"].set_index(["hotel", "share"])
    share = float(step.args[0]["shapes"][0]["x0"]) / 100
    assert text.startswith(f"топ-{share * 100:.0f} %")
    for hotel in ("все отели", "City Hotel", "Resort Hotel"):
        loss = model.loc[(hotel, share), "total"] / model.loc[(hotel, 0.0), "total"] * 100
        assert f"{hotel} — потери {loss:.0f} %" in text
    assert len(fig.layout.shapes) == 1 and fig.layout.annotations[0].text.startswith("топ-5 %")


def test_chart2_has_restyle_dropdown(tables):
    """Выбор отеля на графике 2 — выпадающий список с restyle, видимость задана для всех трасс."""
    fig = fig_top_share(tables["curves"])
    for button in fig.layout.updatemenus[0].buttons:
        assert button.method == "restyle" and len(button.args[0]["visible"]) == len(fig.data)


def test_slider_text_has_best_share_per_hotel(tables):
    """Вывод к графику 2 называет долю с минимальными потерями, посчитанную по входной кривой, для каждого отеля."""
    curves = tables["curves"]
    text = _slider_text(curves)
    model = curves[curves["method"] == "модель"]
    for hotel, rows in model.groupby("hotel"):
        best = rows.loc[rows["total"].idxmin()]
        assert f"{hotel} — {best['share'] * 100:.0f} % (задето {best['touched_reliable'] * 100:.0f} %" in text
    assert "легенд" not in text


def test_risk_text_names_true_argmax(tables):
    """Вывод к графику 4 называет сегмент и месяц с наибольшим риском, взвешенным по броням (пересчёт вручную)."""
    risk = tables["risk"]
    text = _risk_text(risk, min_bookings=50)
    pooled = risk[risk["hotel"] == "все отели"]
    for column, label in (("market_segment", "сегмент"), ("arrival_month", "месяц")):
        mean = {key: np.average(g["mean_proba"], weights=g["bookings"]) for key, g in pooled.groupby(column)
                if g["bookings"].sum() >= 50}
        best = max(mean, key=mean.get)
        expected = f"«{best}»" if label == "сегмент" else f"«{MONTH_NAMES[best]}»"
        assert f"{expected} ({mean[best] * 100:.0f} %)" in text


def test_risk_text_min_bookings_filter(tables):
    """Группы меньше порога не участвуют: при высоком пороге остаются только самые крупные сегменты."""
    risk = tables["risk"]
    sizes = risk[risk["hotel"] == "все отели"].groupby("market_segment")["bookings"].sum()
    big = sizes.idxmax()
    assert f"«{big}»" in _risk_text(risk, min_bookings=int(sizes.max()))


def test_risk_months_in_calendar_order(tables):
    """Ось месяцев графика 4 идёт по календарю, а не по порядку появления."""
    fig = fig_risk(tables["risk"])
    assert list(fig.layout.xaxis.categoryarray) == [MONTH_NAMES[i] for i in range(1, 13)]


def test_assumptions_mention_every_walk_cost():
    """Допущения упоминают каждое значение WALK_COSTS с десятичной запятой."""
    html = _assumptions_html()
    for cost in WALK_COSTS:
        assert str(cost).replace(".", ",") in html


def test_overbooking_text_decimal_comma_only_in_numbers(tables):
    """Десятичная запятая только в числах: конец предложения остаётся точкой, а не «, »."""
    summary = _summary(tables["policies"], tables["curves"], outcome_table(tables["dates"]))
    text = _texts(summary, tables["dates"], tables["curves"], tables["risk"])["overbooking"]
    pooled = tables["dates"][tables["dates"]["hotel"] == "все отели"]
    mean_expected = f"{pooled['expected'].mean():.1f}".replace(".", ",")
    mean_actual = f"{pooled['actual'].mean():.1f}".replace(".", ",")
    assert text.startswith(f"В среднем на дату ожидалось {mean_expected} отмен, случилось {mean_actual}. Осторожная")
