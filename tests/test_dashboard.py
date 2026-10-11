"""Тесты эталонных расчётов дашборда."""
import numpy as np
import pandas as pd
import pytest

from src.dashboard import (
    SCORED_COLUMNS, SCORED_TEST_PATH, comma, count, filter_frame, model_threshold, money, percent, scored_frame,
)
from src.evaluation import Assumptions
from src.policies import ALL_HOTELS

FRAME = pd.DataFrame({
    "hotel": ["City Hotel", "City Hotel", "City Hotel", "Resort Hotel", "Resort Hotel", "Resort Hotel"],
    "market_segment": ["Aviation", "Online TA", "Direct", "Online TA", "Direct", "Online TA"],
    "arrival_month": [5, 5, 6, 6, 7, 7],
    "lead_time": [10.0, 200.0, 30.0, 100.0, 5.0, 60.0],
    "value": [100.0, 400.0, 300.0, 200.0, 0.0, 500.0],
    "is_canceled": [0, 1, 0, 1, 0, 1],
    "proba": [0.1, 0.8, 0.3, 0.6, 0.05, 0.4],
})
SEGMENTS = ["Aviation", "Direct", "Online TA"]
MONTHS = [5, 6, 7]


def test_filter_frame():
    """Фильтр по отелю, сегментам и месяцам; «все отели» — оба отеля; сегмента нет в отеле — пустой срез."""
    assert len(filter_frame(FRAME, ALL_HOTELS, SEGMENTS, MONTHS)) == 6
    assert filter_frame(FRAME, "City Hotel", SEGMENTS, MONTHS)["hotel"].eq("City Hotel").all()
    assert len(filter_frame(FRAME, ALL_HOTELS, ["Online TA"], [7])) == 1
    assert filter_frame(FRAME, "Resort Hotel", ["Aviation"], MONTHS).empty


def test_model_threshold_formula():
    """Порог = loss_prob / (unsold_share + loss_prob): 0,1 / 0,6 по умолчанию, 0,2 / 0,5 на краю ползунков."""
    assert model_threshold() == pytest.approx(1 / 6)
    assert model_threshold(Assumptions(0.3, 0.2)) == pytest.approx(0.4)
    assert model_threshold(Assumptions(0.7, 0.05)) == pytest.approx(0.05 / 0.75)


def test_scored_frame_columns():
    """Таблица для дашборда: нужные колонки; бронь «без возврата» ничего не стоит при отмене; путь — от корня."""
    df = pd.DataFrame({"hotel": ["A", "B"], "market_segment": ["Direct", "Groups"], "arrival_month": [5, 6],
                       "lead_time": [10, 20], "booking_value": [100.0, 50.0],
                       "deposit_type": ["Non Refund", "No Deposit"], "is_canceled": [1, 0]})
    frame = scored_frame(df, np.array([0.7, 0.2]))
    assert frame.columns.tolist() == SCORED_COLUMNS
    assert frame["value"].tolist() == [0.0, 50.0]
    assert SCORED_TEST_PATH.is_absolute()


def test_formatters():
    """Деньги от 0,1 млн € — в миллионах, меньше — в тысячах; запятая; пробел между тысячами; без «-0»."""
    assert money(2_109_480) == "2,11 млн €"
    assert money(23_400) == "23,4 тыс. €"
    assert money(-40) == "0,0 тыс. €"
    assert comma(0.1666) == "0,17"
    assert count(4898) == "4 898"
    assert percent(0.604) == "60 %"
