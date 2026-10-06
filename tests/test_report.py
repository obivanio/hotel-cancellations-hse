"""Тесты интерактивного отчёта."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.overbooking import date_table
from src.policies import equal_share_curves, policy_table, scored
from src.report import SECTIONS, build_report, fig_overbooking, fig_risk, fig_share_curve, fig_top_share, risk_table


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
        for menu in fig.layout.updatemenus:
            assert all(len(button.args[0]["visible"]) == len(fig.data) for button in menu.buttons)
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
