"""Тесты эталонных расчётов дашборда."""
import numpy as np
import pandas as pd
import pytest

from src.dashboard import (
    SCORED_COLUMNS, SCORED_TEST_PATH, comma, count, cut_size, filter_frame, key_numbers, model_threshold, money,
    overbooking_totals, percent, queue_buckets, queue_outcome, risk_tables, scored_frame, share_curves,
    threshold_pct, to_scored, verdict_numbers, weekly_overbooking,
)
from src.evaluation import Assumptions
from src.policies import ALL_HOTELS, policy_metrics, top_share_mask

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
EMPTY = FRAME.iloc[0:0]
DATES = pd.DataFrame({
    "hotel": [ALL_HOTELS] * 3 + ["A"] * 3 + ["B"] * 3,
    "arrival_date": ["2017-05-01", "2017-05-02", "2017-05-08"] * 3,
    "actual": [50, 50, 50, 2, 3, 3, 1, 2, 4],
    "cautious": [0, 0, 0, 1, 5, 2, 2, 3, 3],
    "bench_cautious": [0, 0, 0, 1, 1, 4, 1, 2, 3],
    "mean_value": [100.0] * 9,
})


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


def test_cut_size_rounds_half_up():
    """Число броней в первых pct %: целочисленно, половина округляется вверх; 0 % — никого, 100 % — все."""
    assert cut_size(6, 50) == 3
    assert cut_size(6, 25) == 2
    assert cut_size(10, 5) == 1
    assert cut_size(17351, 0) == 0 and cut_size(17351, 100) == 17351


@pytest.mark.parametrize("assumptions", [Assumptions(), Assumptions(0.3, 0.2), Assumptions(0.7, 0.05)])
@pytest.mark.parametrize("pct", [0, 33, 50, 100])
def test_queue_outcome_matches_policy_metrics(assumptions, pct):
    """Итог черты совпадает с policy_metrics для маски «первые cut_size броней по риску» (как top_share_mask)."""
    outcome = queue_outcome(FRAME, pct, assumptions)
    s = to_scored(FRAME)
    expected = policy_metrics(s, top_share_mask(s.proba, cut_size(6, pct) / 6), assumptions)
    assert outcome["total"] == pytest.approx(expected["total"])
    assert outcome["touched_share"] == pytest.approx(expected["touched_reliable"])
    assert outcome["caught_share"] == pytest.approx(expected["caught_cancels"])


def test_queue_outcome_manual():
    """Предоплата с половины: брони риска 0,8, 0,6, 0,4 — все три отмены пойманы, задетых и пропущенных нет."""
    outcome = queue_outcome(FRAME, 50)
    assert outcome["total"] == pytest.approx(0.0)
    assert (outcome["caught"], outcome["touched"], outcome["missed_cancels"]) == (3, 0, 0)
    nobody = queue_outcome(FRAME, 0)
    assert nobody["total"] == pytest.approx(550.0) and nobody["missed_share"] == 1.0


def test_key_numbers():
    """Без предоплаты 550, со всех — 10 % стоимости надёжных броней (400); порог 1/6 — у 4 из 6 броней (67 %)."""
    numbers = key_numbers(FRAME)
    assert numbers["baseline"] == pytest.approx(550.0)
    assert numbers["everyone"] == pytest.approx(40.0)
    assert numbers["threshold_pct"] == 67
    assert threshold_pct(FRAME, Assumptions(0.3, 0.2)) == 50


def test_queue_buckets():
    """Ровно 100 полосок, сумма броней равна срезу, риск по полоскам не растёт, первая полоска — самая рискованная."""
    frame = pd.DataFrame({"proba": np.linspace(0.9, 0.01, 250), "is_canceled": [1, 0] * 125})
    buckets = queue_buckets(frame)
    assert len(buckets) == 100
    assert buckets["bookings"].sum() == 250
    assert buckets["mean_risk"].is_monotonic_decreasing
    assert buckets.loc[0, "mean_risk"] == pytest.approx(np.linspace(0.9, 0.01, 250)[:3].mean())


def test_queue_buckets_small_slice():
    """Шесть броней: шесть непустых полосок по одной брони, остальные пустые — нули, без NaN."""
    buckets = queue_buckets(FRAME)
    assert (buckets["bookings"] > 0).sum() == 6
    assert buckets.loc[buckets["bookings"] > 0, "mean_risk"].tolist() == pytest.approx([0.8, 0.6, 0.4, 0.3, 0.1, 0.05])
    assert not buckets.isna().any().any()


def test_queue_buckets_smooths_equal_risk():
    """Две брони с одинаковым риском (одна отменена) в разных полосках — обе показывают 0,5; разный риск — 0 и 1."""
    same = pd.DataFrame({"proba": [0.5, 0.5], "is_canceled": [1, 0]})
    shares = queue_buckets(same).query("bookings > 0")["cancel_share"]
    assert shares.tolist() == [0.5, 0.5]
    distinct = pd.DataFrame({"proba": [0.6, 0.5], "is_canceled": [1, 0]})
    assert queue_buckets(distinct).query("bookings > 0")["cancel_share"].tolist() == [1.0, 0.0]


def test_empty_slice_no_errors_no_nan():
    """Пустой срез: итог и опорные цифры — нули, полоски пустые, кривые пустые; без исключений и NaN."""
    outcome = queue_outcome(EMPTY, 70)
    numbers = key_numbers(EMPTY)
    assert not any(np.isnan(value) for value in [*outcome.values(), *numbers.values()])
    assert numbers["bookings"] == 0 and numbers["threshold_pct"] == 0
    assert queue_buckets(EMPTY)["bookings"].sum() == 0
    assert share_curves(EMPTY).empty


def test_share_curves_ends():
    """101 точка на метод; при 0 % обе кривые — «без предоплаты», при 100 % — «со всех»."""
    curves = share_curves(FRAME).set_index(["method", "pct"])["total"]
    assert len(curves) == 202
    assert curves.xs(0, level="pct").tolist() == pytest.approx([550.0, 550.0])
    assert curves.xs(100, level="pct").tolist() == pytest.approx([40.0, 40.0])


def test_risk_tables():
    """Сегмент × месяц: средний риск и доля отмен в %, число броней; месяцы — по-русски; пустая клетка — NaN."""
    tables = risk_tables(FRAME)
    assert tables["risk"].loc["Online TA", "июл"] == pytest.approx(40.0)
    assert tables["cancel"].loc["Online TA", "май"] == pytest.approx(100.0)
    assert tables["bookings"].loc["Online TA"].sum() == 3
    assert np.isnan(tables["cancel"].loc["Aviation", "июн"])


def test_weekly_overbooking():
    """Недели пн–вс: 1–2 мая и 8 мая; «все отели» — сумма двух отелей, сводные строки (50 на дату) не берутся."""
    weeks = weekly_overbooking(DATES, ALL_HOTELS)
    assert weeks["label"].tolist() == ["1–2 мая", "8–8 мая"]
    assert weeks["month"].tolist() == [5, 5]
    assert weeks["actual"].tolist() == [8, 7]
    assert weeks["cautious"].tolist() == [11, 5]
    assert weeks["bench_cautious"].tolist() == [5, 7]
    single = weekly_overbooking(DATES, "A")
    assert single["actual"].tolist() == [5, 3] and single["bench_cautious"].tolist() == [2, 4]


def test_overbooking_totals():
    """«Все отели» — шесть строк двух отелей: без нехватки 3 из 6, перепродано 12, без номера 4, на дату 3–8 номеров;
    сводные не в счёт."""
    totals = overbooking_totals(DATES, ALL_HOTELS)
    assert totals["dates"] == 6
    assert totals["covered"] == pytest.approx(0.5)
    assert totals["walked"] == 4.0 and totals["resold"] == 12.0
    assert totals["covered_bench"] == pytest.approx(5 / 6)
    assert (totals["per_day"], totals["per_day_min"], totals["per_day_max"]) == pytest.approx((16 / 3, 3.0, 8.0))
    single = overbooking_totals(DATES, "A")
    assert (single["per_day"], single["per_day_min"], single["per_day_max"]) == pytest.approx((8 / 3, 1.0, 5.0))
    assert (single["dates"], single["resold"], single["walked"]) == (3, 6.0, 2.0)
    assert single["covered"] == pytest.approx(2 / 3)
    assert overbooking_totals(DATES, "нет такого")["covered"] == 0.0


def test_verdict_numbers():
    """Цифры рекомендаций: правила — из строк «все отели», доли дат без нехватки — по отелям; «все» — сумма отелей."""
    policies = pd.DataFrame({
        "hotel": [ALL_HOTELS] * 4,
        "policy": ["как сейчас", "со всех", "модель, порог 0,17", "модель, топ-30 %"],
        "total": [2.0, 0.6, 0.5, 0.9], "prepay_share": [0.0, 1.0, 0.7, 0.3],
        "touched_reliable": [0.0, 1.0, 0.6, 0.2], "caught_cancels": [0.0, 1.0, 0.9, 0.5],
    })
    dates = DATES.replace({"A": "City Hotel", "B": "Resort Hotel"})
    numbers = verdict_numbers(policies, dates)
    assert (numbers["baseline"], numbers["model"], numbers["top30_touched"]) == (2.0, 0.5, 0.2)
    assert numbers["covered_all"] == pytest.approx(0.5)
    assert numbers["covered_city"] == pytest.approx(2 / 3)
    assert numbers["covered_resort"] == pytest.approx(1 / 3)


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
