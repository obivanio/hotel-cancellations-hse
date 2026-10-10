"""Тесты расчётов дашборда."""
import numpy as np
import pandas as pd
import pytest

from src.dashboard import (
    SCORED_COLUMNS, SCORED_TEST_PATH, count, display_rules, filter_frame, key_numbers, model_threshold,
    overbooking_summary, risk_tables, rules_table, scored_frame, share_curves, to_scored,
)
from src.evaluation import Assumptions
from src.policies import ALL_HOTELS, policy_metrics

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
    """Фильтр по отелю, сегментам и месяцам; «все отели» — оба отеля."""
    assert len(filter_frame(FRAME, ALL_HOTELS, SEGMENTS, MONTHS)) == 6
    assert filter_frame(FRAME, "City Hotel", SEGMENTS, MONTHS)["hotel"].eq("City Hotel").all()
    assert len(filter_frame(FRAME, ALL_HOTELS, ["Online TA"], [7])) == 1


def test_filter_segment_absent_in_hotel():
    """Сегмент, которого нет в выбранном отеле (Aviation в курортном), даёт пустой срез, а не ошибку."""
    assert filter_frame(FRAME, "Resort Hotel", ["Aviation"], MONTHS).empty


def test_model_threshold_formula():
    """Порог = loss_prob / (unsold_share + loss_prob): 0,1 / 0,6 по умолчанию, 0,2 / 0,5 на краю ползунков."""
    assert model_threshold() == pytest.approx(1 / 6)
    assert model_threshold(Assumptions(0.3, 0.2)) == pytest.approx(0.4)
    assert model_threshold(Assumptions(0.7, 0.05)) == pytest.approx(0.05 / 0.75)


@pytest.mark.parametrize("assumptions", [Assumptions(), Assumptions(0.3, 0.2), Assumptions(0.7, 0.05)])
def test_key_numbers_match_policy_metrics(assumptions):
    """Ключевые цифры совпадают с policy_metrics при тех же допущениях и пороге по формуле."""
    numbers = key_numbers(FRAME, assumptions)
    s = to_scored(FRAME)
    model = policy_metrics(s, s.proba >= model_threshold(assumptions), assumptions)
    everyone = policy_metrics(s, np.ones(6, dtype=bool), assumptions)
    assert numbers["baseline"] == pytest.approx(policy_metrics(s, np.zeros(6, dtype=bool), assumptions)["total"])
    assert numbers["model"] == pytest.approx(model["total"])
    assert numbers["saving"] == pytest.approx(everyone["total"] - model["total"])
    assert numbers["touched_reliable"] == pytest.approx(model["touched_reliable"])


def test_key_numbers_manual():
    """Порог 1/6: предоплата с броней риска 0,8, 0,6, 0,3 и 0,4; пропущенных отмен нет, лишняя предоплата — 300."""
    numbers = key_numbers(FRAME)
    assert numbers["baseline"] == pytest.approx(550.0)
    assert numbers["model"] == pytest.approx(30.0)
    assert numbers["everyone"] == pytest.approx(40.0)
    assert numbers["touched_reliable"] == pytest.approx(0.5)


def test_empty_slice_no_errors_no_nan():
    """Пустой срез: ключевые цифры — нули, таблица правил и кривые пустые, без исключений и NaN."""
    empty = filter_frame(FRAME, "Resort Hotel", ["Aviation"], MONTHS)
    numbers = key_numbers(empty)
    assert numbers["bookings"] == 0
    assert not any(np.isnan(value) for value in numbers.values())
    assert rules_table(empty, lead_days=7, share=0.5).empty
    assert share_curves(empty).empty


def test_rules_table_five_rules():
    """Пять правил среза; экономия «как сейчас» — 0; топ-50 % — ровно половина броней; запятая и пробелы в числах."""
    table = rules_table(FRAME, lead_days=50, share=0.5)
    assert table["policy"].tolist() == ["как сейчас", "со всех", "срок брони > 50 дн.", "модель, порог 0,17",
                                        "модель, топ-50 %"]
    assert table.loc[0, "saving"] == 0.0
    assert table.loc[4, "prepay_share"] == 0.5
    shown = display_rules(table)
    assert shown.loc[4, "предоплата, % броней"] == "50 %"
    assert shown.loc[0, "потери, млн €"] == "0,001"
    assert count(4898) == "4 898"


def test_share_curves_ends():
    """При доле 0 обе кривые — «как сейчас», при доле 1 — «со всех»."""
    curves = share_curves(FRAME, shares=np.array([0.0, 1.0])).set_index(["method", "share"])["total"]
    assert curves.xs(0.0, level="share").tolist() == pytest.approx([550.0, 550.0])
    assert curves.xs(1.0, level="share").tolist() == pytest.approx([40.0, 40.0])


def test_overbooking_summary_hotels_only():
    """«Все отели» складывает строки отелей (не сводные), фильтр месяцев работает, пустой срез — нули."""
    dates = pd.DataFrame({
        "hotel": [ALL_HOTELS, "A", "B", "A"],
        "arrival_date": ["2017-05-01", "2017-05-01", "2017-05-01", "2017-06-01"],
        "actual": [5, 2, 3, 1], "cautious": [4, 1, 3, 2], "bench_cautious": [2, 1, 1, 0],
        "mean_value": [100.0, 100.0, 100.0, 100.0],
    })
    both = overbooking_summary(dates, ALL_HOTELS, [5, 6])
    assert both["модель"]["resold"] == 5.0 and both["модель"]["walked"] == 1.0
    assert overbooking_summary(dates, "A", [5])["модель"]["resold"] == 1.0
    assert overbooking_summary(dates, "A", [8])["модель"]["net"] == 0.0


def test_risk_tables():
    """Сегмент × месяц: средний риск и доля отмен в %, число броней; месяцы — по-русски."""
    tables = risk_tables(FRAME)
    assert tables["risk"].loc["Online TA", "июл"] == pytest.approx(40.0)
    assert tables["cancel"].loc["Online TA", "май"] == pytest.approx(100.0)
    assert tables["bookings"].loc["Online TA"].sum() == 3


def test_scored_frame_columns():
    """Таблица для дашборда: нужные колонки; бронь «без возврата» ничего не стоит при отмене; путь — от корня."""
    df = pd.DataFrame({"hotel": ["A", "B"], "market_segment": ["Direct", "Groups"], "arrival_month": [5, 6],
                       "lead_time": [10, 20], "booking_value": [100.0, 50.0],
                       "deposit_type": ["Non Refund", "No Deposit"], "is_canceled": [1, 0]})
    frame = scored_frame(df, np.array([0.7, 0.2]))
    assert frame.columns.tolist() == SCORED_COLUMNS
    assert frame["value"].tolist() == [0.0, 50.0]
    assert SCORED_TEST_PATH.is_absolute()
