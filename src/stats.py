"""Статистические тесты и размеры эффекта для проверки гипотез."""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import expected_freq

ALPHA = 0.05
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class StatResult:
    """Результат теста: статистика, p-value и размер эффекта с доверительным интервалом."""

    test: str
    statistic: float
    p_value: float
    effect_name: str
    effect: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")


def chi2_independence(groups: pd.Series, outcome: pd.Series) -> StatResult:
    """χ² на независимость группы и исхода; размер эффекта — V Крамера."""
    table = pd.crosstab(groups, outcome)
    chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return StatResult("χ² независимости", float(chi2), float(p_value), "V Крамера", float(np.sqrt(chi2 / (n * k))))


def min_expected_count(groups: pd.Series, outcome: pd.Series) -> float:
    """Минимальная ожидаемая частота в таблице сопряжённости (допущение χ²: не меньше 5)."""
    return float(expected_freq(pd.crosstab(groups, outcome)).min())


def two_proportion_ztest(success_a: int, n_a: int, success_b: int, n_b: int) -> StatResult:
    """z-тест для двух долей; эффект — разница долей a − b с 95 % доверительным интервалом."""
    p_a, p_b = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    diff = p_a - p_b
    z = diff / se_pooled if se_pooled > 0 else 0.0
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    return StatResult(
        "z-тест для двух долей", float(z), float(2 * stats.norm.sf(abs(z))), "разница долей", float(diff),
        float(diff - Z_95 * se_diff), float(diff + Z_95 * se_diff),
    )


def spearman(x: pd.Series, y: pd.Series) -> StatResult:
    """Ранговая корреляция Спирмена — есть ли монотонный тренд."""
    rho, p_value = stats.spearmanr(x, y)
    return StatResult("корреляция Спирмена", float(rho), float(p_value), "ρ Спирмена", float(rho))


def kruskal_wallis(values: pd.Series, groups: pd.Series) -> StatResult:
    """Краскел — Уоллис: различаются ли распределения по группам; эффект — ε² = H / (n − 1)."""
    samples = [values[groups == group].to_numpy() for group in sorted(groups.unique())]
    h_stat, p_value = stats.kruskal(*samples)
    return StatResult("Краскел — Уоллис", float(h_stat), float(p_value), "ε²", float(h_stat / (len(values) - 1)))


def holm(p_values: Sequence[float]) -> list[float]:
    """Поправка Холма на множественные сравнения: скорректированные p-value в исходном порядке."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()
