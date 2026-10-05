"""Тесты статистических функций."""
import pandas as pd
import pytest

from src.stats import (
    chi2_independence, holm, kruskal_wallis, min_expected_count, spearman, two_proportion_ztest,
)


def test_ztest_equal_proportions():
    """Одинаковые доли: p = 1, разница 0, доверительный интервал накрывает 0."""
    res = two_proportion_ztest(50, 100, 50, 100)
    assert res.p_value == pytest.approx(1.0)
    assert res.effect == pytest.approx(0.0)
    assert res.ci_low < 0 < res.ci_high


def test_ztest_different_proportions():
    """Сильно разные доли: p < 0,001, разница 0,8 внутри доверительного интервала."""
    res = two_proportion_ztest(90, 100, 10, 100)
    assert res.p_value < 0.001
    assert res.effect == pytest.approx(0.8)
    assert res.ci_low < 0.8 < res.ci_high


def test_chi2_no_association():
    """Одинаковое распределение исхода в группах: V Крамера = 0, p = 1."""
    groups = pd.Series(["a"] * 10 + ["b"] * 10)
    outcome = pd.Series([0, 1] * 10)
    res = chi2_independence(groups, outcome)
    assert res.effect == pytest.approx(0.0)
    assert res.p_value == pytest.approx(1.0)


def test_chi2_full_association():
    """Исход полностью определяется группой: V Крамера = 1, p < 0,001."""
    groups = pd.Series(["a"] * 20 + ["b"] * 20)
    outcome = pd.Series([1] * 20 + [0] * 20)
    res = chi2_independence(groups, outcome)
    assert res.effect == pytest.approx(1.0)
    assert res.p_value < 0.001


def test_min_expected_count():
    """Минимальная ожидаемая частота считается по маргиналам таблицы сопряжённости."""
    groups = pd.Series(["a"] * 10 + ["b"] * 10)
    outcome = pd.Series([0, 1] * 10)
    assert min_expected_count(groups, outcome) == pytest.approx(5.0)


def test_spearman_monotonic():
    """Строго возрастающая зависимость: ρ = 1."""
    x = pd.Series(range(20))
    assert spearman(x, x * 2).effect == pytest.approx(1.0)


def test_kruskal_same_and_shifted():
    """Одинаковые группы — p = 1 и ε² = 0; сдвинутые — p < 0,01 и ε² > 0,5."""
    groups = pd.Series(["a"] * 10 + ["b"] * 10)
    same = kruskal_wallis(pd.Series(list(range(10)) * 2), groups)
    assert same.p_value == pytest.approx(1.0)
    assert same.effect == pytest.approx(0.0)
    shifted = kruskal_wallis(pd.Series(list(range(10)) + list(range(100, 110))), groups)
    assert shifted.p_value < 0.01
    assert 0.5 < shifted.effect <= 1.0


def test_holm_manual_example():
    """Поправка Холма: [0.01, 0.04, 0.03] → [0.03, 0.06, 0.06]."""
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    """Скорректированные p-value не превышают 1."""
    assert holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])
