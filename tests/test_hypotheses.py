"""Тесты гипотез H1–H6 на синтетических данных с эффектом и без него."""
import numpy as np
import pandas as pd

from src.data import load_raw
from src.hypotheses import STATEMENTS, h2_pairwise, h6_seasonality, hypotheses_table, reject_h0, robustness
from src.hypotheses import save_hypotheses, segments

SEGMENTS = ["Online TA", "Direct", "Groups", "Corporate", "Aviation"]


def _bookings(effect: bool, n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Синтетические брони: с заложенными эффектами H1–H6 или без них."""
    rng = np.random.default_rng(seed)
    lead = rng.integers(0, 500, n)
    city = rng.random(n) < 0.5
    repeated = rng.random(n) < 0.2
    requests = rng.integers(0, 3, n)
    segment = rng.choice(SEGMENTS, n)
    month = rng.integers(1, 13, n)
    if effect:
        prob = (0.1 + 0.5 * lead / 500 + 0.1 * city - 0.1 * repeated - 0.1 * (requests > 0)
                + 0.15 * (segment == "Online TA"))
        adr = np.where(city, 100 + 2 * month, 40 + 12 * month) + rng.normal(0, 10, n)
    else:
        prob = np.full(n, 0.3)
        adr = 100 + rng.normal(0, 10, n)
    return pd.DataFrame({
        "is_canceled": (rng.random(n) < np.clip(prob, 0.01, 0.99)).astype(int),
        "lead_time": lead,
        "hotel": np.where(city, "City Hotel", "Resort Hotel"),
        "is_repeated_guest": repeated.astype(int),
        "total_of_special_requests": requests,
        "market_segment": segment,
        "arrival_month": month,
        "adr": adr,
        "is_free": 0,
        "n_rooms": 1,
        "deposit_type": "No Deposit",
    })


def test_all_hypotheses_detected_with_effect():
    """Заложенные эффекты H1–H6 обнаруживаются после поправки Холма."""
    table = hypotheses_table(_bookings(effect=True))
    assert table["reject_h0"].all(), table[["hypothesis", "p_holm", "effect"]]


def test_no_hypothesis_rejected_without_effect():
    """Без эффектов ни одна нулевая гипотеза не отвергается."""
    table = hypotheses_table(_bookings(effect=False))
    assert not table["reject_h0"].any(), table[["hypothesis", "p_holm", "effect"]]


def test_h6_bootstrap_ci_with_effect():
    """С заложенной сезонностью курорта бутстреп-ДИ разницы ε² целиком выше нуля и содержит оценку."""
    res = h6_seasonality(_bookings(effect=True))
    assert 0 < res.ci_low <= res.effect <= res.ci_high


def test_h6_bootstrap_ci_reproducible():
    """Бутстреп с фиксированным зерном даёт одинаковый ДИ при повторном запуске."""
    df = _bookings(effect=True)
    first, second = h6_seasonality(df), h6_seasonality(df)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_reject_h0_rule():
    """H0 отвергается только при p < 0,05, нужном знаке и ДИ, не накрывающем 0 (пустой ДИ не мешает)."""
    nan = float("nan")
    p_holm = pd.Series([0.001, 0.001, 0.001, 0.2, 0.001, 0.001])
    effect = pd.Series([0.3, 0.3, -0.3, 0.3, 0.3, -0.2])
    expected = pd.Series([1, 1, 1, 1, 1, -1])
    ci_low = pd.Series([nan, -0.1, nan, nan, 0.1, -0.3])
    ci_high = pd.Series([nan, 0.5, nan, nan, 0.5, -0.1])
    assert reject_h0(p_holm, effect, expected, ci_low, ci_high).tolist() == [True, False, False, False, True, True]


def test_table_shape():
    """Таблица: 6 гипотез по порядку, p-value в [0, 1], поправленный p не меньше исходного."""
    table = hypotheses_table(_bookings(effect=True))
    assert table["hypothesis"].tolist() == list(STATEMENTS)
    assert table["p_value"].between(0, 1).all()
    assert (table["p_holm"] >= table["p_value"]).all()


def test_segments_groups_small():
    """Малые сегменты Aviation и Complementary объединяются в «Прочие»."""
    df = pd.DataFrame({"market_segment": ["Aviation", "Complementary", "Direct"]})
    assert segments(df).tolist() == ["Прочие", "Прочие", "Direct"]


def test_h2_pairwise():
    """Попарные сравнения сегментов: все 10 пар из 5 групп, поправка Холма."""
    pairs = h2_pairwise(_bookings(effect=True))
    assert len(pairs) == 10
    assert (pairs["p_holm"] >= pairs["p_value"]).all()


def test_save_hypotheses(tmp_path):
    """Таблица с отметкой устойчивости сохраняется в CSV."""
    table = hypotheses_table(_bookings(effect=True))
    decisions = pd.DataFrame({"robust": True}, index=table["hypothesis"])
    path = tmp_path / "hypotheses.csv"
    save_hypotheses(table, decisions, path)
    assert pd.read_csv(path)["robust"].all()


def test_real_data_robustness():
    """Интеграционный: на реальных данных перепроверка строится для 6 гипотез и 4 вариантов."""
    decisions = robustness(load_raw())
    assert decisions.shape == (6, 5)
    assert list(decisions.columns) == ["C", "A", "B", "C без Non Refund", "robust"]
