"""Тесты овербукинга."""
import numpy as np
import pandas as pd
import pytest

from src.overbooking import (
    benchmark_proba, benchmark_rates, cancel_distribution, cautious_count, date_table, outcome_table,
    overbooking_outcome,
)
from src.policies import ALL_HOTELS


def test_cancel_distribution_manual():
    """Две брони с риском 0,5 и 0,2: P(0) = 0,4, P(1) = 0,5, P(2) = 0,1; три брони — сумма вероятностей 1."""
    assert cancel_distribution(np.array([0.5, 0.2])) == pytest.approx([0.4, 0.5, 0.1])
    dist = cancel_distribution(np.array([0.5, 0.2, 0.9]))
    assert len(dist) == 4 and dist.sum() == pytest.approx(1.0)
    assert dist[3] == pytest.approx(0.5 * 0.2 * 0.9)


def test_cautious_count_manual():
    """P(отмен ≥ 1) = 0,6: при уверенности 90 % рассчитываем на 0 отмен, при 50 % — на 1."""
    proba = np.array([0.5, 0.2])
    assert cautious_count(proba, 0.9) == 0
    assert cautious_count(proba, 0.5) == 1


def test_cautious_count_certain():
    """Вероятности 0 и 1 (бывают после калибровки) и граница ровно 90 % считаются без ошибок округления."""
    assert cautious_count(np.array([1.0, 1.0, 0.0])) == 2
    assert cautious_count(np.array([0.9])) == 1
    assert cautious_count(np.array([])) == 0


def test_benchmark_new_cell():
    """Бенчмарк «месяц × день недели»: доля отмен ячейки обучения; новая ячейка — общая доля, без NaN."""
    train = pd.DataFrame({"arrival_month": [5, 5, 6], "arrival_weekday": [0, 0, 1], "is_canceled": [1, 0, 1]})
    rates = benchmark_rates(train)
    new = pd.DataFrame({"arrival_month": [5, 7], "arrival_weekday": [0, 3]})
    assert benchmark_proba(rates, new, fallback=0.3).tolist() == pytest.approx([0.5, 0.3])


def _bookings() -> pd.DataFrame:
    """Четыре брони: два отеля, две даты заезда."""
    return pd.DataFrame({
        "hotel": ["A", "A", "B", "B"],
        "arrival_date": pd.to_datetime(["2017-05-01", "2017-05-01", "2017-05-01", "2017-05-02"]),
        "is_canceled": [1, 0, 1, 1],
        "booking_value": [100.0, 300.0, 50.0, 70.0],
    })


def test_date_table():
    """По датам: строки «все отели» и каждого отеля; ожидание — сумма вероятностей, средняя стоимость брони."""
    table = date_table(_bookings(), np.array([0.5, 0.5, 0.9, 1.0]), np.full(4, 0.3))
    pooled = table[table["hotel"] == ALL_HOTELS].set_index("arrival_date")
    assert len(table) == 2 + 3
    assert pooled.loc["2017-05-01", "bookings"] == 3 and pooled.loc["2017-05-01", "actual"] == 2
    assert pooled.loc["2017-05-01", "expected"] == pytest.approx(1.9)
    assert pooled.loc["2017-05-01", "mean_value"] == pytest.approx(150.0)
    assert table["bench_expected"].notna().all()


def test_overbooking_outcome_manual():
    """План 3 и 1 номера при 2 и 2 отменах: перепродано 2 + 1, одному гостю не хватило номера в первый день."""
    table = pd.DataFrame({"plan": [3, 1], "actual": [2, 2], "mean_value": [100.0, 50.0]})
    outcome = overbooking_outcome(table, "plan", walk_cost=1.0)
    assert outcome == pytest.approx({"resold": 3.0, "short_dates": 1.0, "walked": 1.0, "gain": 250.0,
                                     "walk_cost": 100.0, "net": 150.0})
    assert overbooking_outcome(table, "plan", walk_cost=2.0)["net"] == pytest.approx(50.0)


def test_outcome_table_hotels_only():
    """Итог считается по отелям: сводные строки «все отели» не удваивают перепроданные номера."""
    table = date_table(_bookings(), np.array([1.0, 0.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0, 1.0]))
    outcomes = outcome_table(table, walk_costs=(1.0,)).set_index("plan")
    assert outcomes.loc["модель", "resold"] == 3.0
    assert outcomes.loc["бенчмарк", "walked"] == 0.0
