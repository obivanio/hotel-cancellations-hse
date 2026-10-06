"""Тесты метрик и порога по деньгам."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    Assumptions, best_threshold, calibration_table, loss_curve, loss_sensitivity, losses, score_table,
    threshold_report, threshold_sensitivity, value_at_risk,
)

Y = pd.Series([1, 1, 0, 0])
PROBA = np.array([0.9, 0.3, 0.6, 0.1])
VALUE = np.array([100.0, 200.0, 300.0, 400.0])


def test_losses_manual():
    """Порог 0,5: пропущена отмена на 200 (× 0,5), лишняя предоплата на 300 (× 0,1)."""
    assert losses(Y, PROBA, VALUE, 0.5) == pytest.approx({"missed": 100.0, "false_alarm": 30.0, "total": 130.0})


def test_losses_extremes():
    """Без предоплаты теряем половину отменённых; со всех — 10 % стоимости надёжных."""
    assert losses(Y, PROBA, VALUE, np.inf)["total"] == pytest.approx(150.0)
    assert losses(Y, PROBA, VALUE, 0.0)["total"] == pytest.approx(70.0)


def test_best_threshold():
    """Оптимум — порог, при котором предоплату берут с обеих отмен и не трогают бронь с 0,1."""
    curve = loss_curve(Y, PROBA, VALUE)
    assert best_threshold(curve) == pytest.approx(0.11)
    assert curve["total"].min() == pytest.approx(30.0)


def test_value_at_risk_non_refund():
    """Отмена брони «без возврата» ничего не стоит отелю."""
    df = pd.DataFrame({"booking_value": [100.0, 50.0], "deposit_type": ["Non Refund", "No Deposit"]})
    assert value_at_risk(df).tolist() == [0.0, 50.0]


def test_score_table_and_report():
    """Метрики в [0, 1]; отчёт при пороге считает точность, полноту и долю предоплат."""
    table = score_table(Y, {"model": PROBA})
    assert table.loc["model"].between(0, 1).all()
    report = threshold_report(Y, PROBA, 0.5)
    assert report == pytest.approx({"precision": 0.5, "recall": 0.5, "prepay_share": 0.5})


def test_sensitivity_grows_with_loss_prob():
    """Чем дороже отпугнуть гостя, тем выше оптимальный порог."""
    rng = np.random.default_rng(0)
    proba = rng.random(2000)
    y = pd.Series((rng.random(2000) < proba).astype(int))
    table = threshold_sensitivity(y, proba, np.full(2000, 100.0), [0.5], [0.05, 0.2])
    assert table.loc[0.5, 0.05] < table.loc[0.5, 0.2]


def test_loss_sensitivity_is_minimum_of_curve():
    """Каждая клетка — минимум суммарных потерь по порогам при своих допущениях; дороже ошибки — больше потерь."""
    table = loss_sensitivity(Y, PROBA, VALUE, [0.3, 0.7], [0.05, 0.2])
    for unsold in (0.3, 0.7):
        for loss_prob in (0.05, 0.2):
            curve = loss_curve(Y, PROBA, VALUE, Assumptions(unsold, loss_prob))
            assert table.loc[unsold, loss_prob] == pytest.approx(curve["total"].min())
    assert table.loc[0.3, 0.05] <= table.loc[0.7, 0.05] <= table.loc[0.7, 0.2]


def test_calibration_table():
    """Калибровочная таблица: прогноз и наблюдаемая доля в [0, 1]."""
    rng = np.random.default_rng(0)
    proba = rng.random(1000)
    y = pd.Series((rng.random(1000) < proba).astype(int))
    table = calibration_table(y, proba, bins=5)
    assert len(table) == 5
    assert table.stack().between(0, 1).all()
