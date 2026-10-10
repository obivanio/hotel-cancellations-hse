"""Тесты интерактивного отчёта."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.overbooking import WALK_COSTS, date_table, outcome_table, week_labels, weekly_table
from src.policies import equal_share_curves, policy_table, scored
from src.report import (MONTH_NAMES, SECTIONS, _assumptions_html, _int, _key_numbers, _risk_text, _slider_text,
                        _summary, _texts, build_report, fig_overbooking, fig_risk, fig_share_curve, fig_top_share,
                        risk_table)


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
        buttons = [button for menu in fig.layout.updatemenus for button in menu.buttons]
        assert [button.label for button in buttons] == ["все отели", "City Hotel", "Resort Hotel"]
        for button in buttons:
            assert button.method in {"restyle", "update"}
            assert len(button.args[0]["visible"]) == len(fig.data) and any(button.args[0]["visible"])
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
    assert "без предоплаты.</p><p>Основную экономию" in html
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


def test_chart2_legend_and_hover_for_every_hotel(tables):
    """На графике 2 у видимых линий любого отеля есть легенда, а подсказка называет показатель."""
    fig = fig_top_share(tables["curves"])
    for button in fig.layout.updatemenus[0].buttons:
        shown = [trace for trace, visible in zip(fig.data, button.args[0]["visible"]) if visible]
        assert len(shown) == 3 and all(trace.showlegend for trace in shown)
        assert all(trace.hovertemplate.startswith(trace.name) for trace in shown)


def test_curve_text_names_validation_rule_and_baseline(tables):
    """Вывод к графику 1: рабочее правило — порог с валидации; «со всех» и вклад самой предоплаты названы числами."""
    policies = tables["policies"]
    summary = _summary(policies, tables["curves"], outcome_table(tables["dates"]))
    text = _texts(summary, tables["dates"], tables["curves"], tables["risk"])["curve"]
    total = policies[policies["hotel"] == "все отели"].set_index("policy")
    everyone = total.loc["со всех", "total"]
    assert "«модель, порог 0,17»" in text and "подобран на валидации" in text
    assert f"«со всех» теряет {everyone / 1e6:.2f}".replace(".", ",") in text
    assert "сама предоплата" in text
    keys = _key_numbers(summary)
    assert keys["со всех"].startswith(f"{everyone / 1e6:.2f}".replace(".", ","))


def test_slider_text_marks_hindsight(tables):
    """Вывод к графику 2 оговаривает, что минимум найден на тесте задним числом."""
    assert "задним числом" in _slider_text(tables["curves"])


def test_thousands_separator():
    """Целые числа в отчёте — с пробелом между тысячами: 4 898."""
    assert _int(4898.0) == "4 898" and _int(90) == "90"


def test_assumptions_forbid_summing():
    """Допущения предупреждают, что деньги предоплаты и овербукинга не складываются."""
    assert "не складываются" in _assumptions_html()


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
    weeks = weekly_table(tables["dates"])
    error = f"{(weeks['expected'] - weeks['actual']).abs().mean():.1f}".replace(".", ",")
    assert f"Модель в среднем ошибалась на {error} отмены за неделю" in text
    assert "\n\nОсторожная оценка модели перепродаёт" in text and "отель × дата" in text
    assert "по каждому дню" not in text and ", ." not in text


def test_overbooking_chart_is_weekly(tables):
    """График 3: на отель столбики фактов и две линии; подписи оси — недели вроде «3–7 мая»."""
    fig = fig_overbooking(tables["dates"])
    assert [type(t).__name__ for t in fig.data] == ["Bar", "Scatter", "Scatter"] * 3
    assert list(fig.data[0].x) == week_labels(weekly_table(tables["dates"]))
    assert all("–" in label for label in fig.data[0].x)
    assert fig.layout.xaxis.title.text == "неделя заезда" and fig.layout.yaxis.title.text == "отмен за неделю"
    assert len(fig.layout.updatemenus[0].buttons) == 3
