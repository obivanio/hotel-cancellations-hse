"""Тесты правил предоплаты."""
import numpy as np
import pandas as pd
import pytest

from src.policies import (
    ALL_HOTELS, BASELINE, Scored, best_lead_days, equal_share_curves, policy_masks, policy_metrics,
    policy_sensitivity, policy_table, scored, top_share_mask,
)

S = Scored(
    canceled=np.array([True, True, False, False]),
    proba=np.array([0.9, 0.3, 0.6, 0.1]),
    lead_time=np.array([200.0, 50.0, 10.0, 1.0]),
    value=np.array([100.0, 200.0, 300.0, 400.0]),
    hotel=np.array(["A", "A", "B", "B"]),
)


def test_policy_metrics_manual():
    """Предоплата с 1-й и 3-й брони: пропущена отмена на 200, зря задет гость на 300; по половине каждой группы."""
    metrics = policy_metrics(S, np.array([True, False, True, False]))
    assert metrics == pytest.approx({"missed": 100.0, "false_alarm": 30.0, "total": 130.0, "prepay_share": 0.5,
                                     "touched_reliable": 0.5, "caught_cancels": 0.5})


def test_policy_metrics_extremes():
    """«Как сейчас» никого не задевает и теряет половину отменённых; «со всех» задевает всех надёжных."""
    nobody = policy_metrics(S, np.zeros(4, dtype=bool))
    everyone = policy_metrics(S, np.ones(4, dtype=bool))
    assert nobody["total"] == pytest.approx(150.0) and nobody["touched_reliable"] == 0.0
    assert everyone["total"] == pytest.approx(70.0) and everyone["touched_reliable"] == 1.0
    assert everyone["caught_cancels"] == 1.0


def test_policy_metrics_no_cancels():
    """В части без отмен (или без надёжных гостей) доли равны 0, а не NaN."""
    part = S.subset(np.array([False, False, True, True]))
    assert policy_metrics(part, np.ones(2, dtype=bool))["caught_cancels"] == 0.0
    part = S.subset(np.array([True, True, False, False]))
    assert policy_metrics(part, np.ones(2, dtype=bool))["touched_reliable"] == 0.0


def test_top_share_mask_ties():
    """Топ-X % берёт ровно X % броней даже при равных скорах; из равных — те, что раньше в таблице."""
    assert top_share_mask(np.full(10, 0.5), 0.3).tolist() == [True] * 3 + [False] * 7
    assert top_share_mask(np.array([0.1, 0.5, 0.5, 0.9, 0.5]), 0.4).tolist() == [False, True, False, True, False]
    assert top_share_mask(np.array([0.2, 0.1]), 0.0).sum() == 0


def test_best_lead_days_manual():
    """Правило «срок > N»: отменяют только ранние брони; лучший N — наименьший, отделяющий их от поздних."""
    s = Scored(np.array([False, False, True, True]), np.zeros(4), np.array([1.0, 5.0, 10.0, 20.0]),
               np.full(4, 100.0), np.array(["A"] * 4))
    assert best_lead_days(s) == 5


def test_policy_masks_names():
    """Пять правил: как сейчас, со всех, срок, порог модели и топ-X % для каждой доли."""
    masks = policy_masks(S, lead_days=5, threshold=0.5, top_shares=np.array([0.5]))
    assert list(masks) == [BASELINE, "со всех", "срок брони > 5 дн.", "модель, порог 0,50", "модель, топ-50 %"]
    assert masks["модель, порог 0,50"].tolist() == [True, False, True, False]


def test_policy_table_by_hotel():
    """Таблица: все отели и каждый отель; экономия «как сейчас» — 0; топ-50 % — ровно половина броней отеля."""
    table = policy_table(S, lead_days=5, threshold=0.5)
    assert set(table["hotel"]) == {ALL_HOTELS, "A", "B"}
    assert (table.loc[table["policy"] == BASELINE, "saving"] == 0).all()
    assert (table.loc[table["policy"] == "модель, топ-50 %", "prepay_share"] == 0.5).all()


def test_equal_share_curves():
    """Кривые «доля предоплаты → потери»: при доле 0 — «как сейчас», при доле 1 — «со всех», для обоих способов."""
    curves = equal_share_curves(S, shares=np.array([0.0, 0.5, 1.0]))
    total = curves[curves["hotel"] == ALL_HOTELS].set_index(["method", "share"])["total"]
    assert set(curves["method"]) == {"модель", "срок брони"}
    assert total.xs(0.0, level="share").tolist() == pytest.approx([150.0, 150.0])
    assert total.xs(1.0, level="share").tolist() == pytest.approx([70.0, 70.0])


def test_policy_sensitivity():
    """Сетка допущений: строка на каждую пару, лучшее правило — одно из четырёх, порог — по формуле."""
    table = policy_sensitivity(S, S, [0.5], [0.1, 0.2])
    assert len(table) == 2
    assert set(table["best"]) <= {BASELINE, "со всех", "срок брони", "модель"}
    assert table["threshold"].tolist() == pytest.approx([0.1 / 0.6, 0.2 / 0.7])
    assert (table["best_total"] == table[["как сейчас", "со всех", "срок брони", "модель"]].min(axis=1)).all()


def test_scored_from_frame():
    """Scored из таблицы броней: отмена — bool, бронь «без возврата» ничего не стоит при отмене."""
    df = pd.DataFrame({"is_canceled": [1, 0], "lead_time": [10, 20], "booking_value": [100.0, 50.0],
                       "deposit_type": ["Non Refund", "No Deposit"], "hotel": ["A", "B"]})
    s = scored(df, np.array([0.7, 0.2]))
    assert s.canceled.tolist() == [True, False]
    assert s.value.tolist() == [0.0, 50.0]
    assert s.hotel.tolist() == ["A", "B"]
