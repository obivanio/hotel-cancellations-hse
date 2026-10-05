"""Тесты признаков."""
import pandas as pd
import pytest

from src.cleaning import clean
from src.features import LEAKAGE_COLUMNS, build_features


@pytest.fixture(name="booking")
def fixture_booking(make_raw):
    """Одна бронь на 2 номера (две одинаковые строки), с ребёнком, после очистки."""
    df, _ = clean(make_raw([{"children": 1.0}, {"children": 1.0}]))
    return df


def test_dates(booking):
    """Дата заезда собирается из колонок, дата брони = заезд − lead_time."""
    row = build_features(booking).iloc[0]
    assert row["arrival_date"] == pd.Timestamp("2016-07-01")
    assert row["booking_date"] == pd.Timestamp("2016-06-01")
    assert row["arrival_month"] == 7
    assert row["arrival_weekday"] == 4
    assert row["season"] == "summer"
    assert row["reservation_status_date"] == pd.Timestamp("2016-07-04")


def test_stay_and_guests(booking):
    """Ночи, доля выходных, гости, семья."""
    row = build_features(booking).iloc[0]
    assert row["nights_week"] == 2
    assert row["nights_weekend"] == 1
    assert row["nights_total"] == 3
    assert row["weekend_share"] == pytest.approx(1 / 3)
    assert row["guests"] == 3
    assert row["is_family"] == 1


def test_money_and_flags(booking):
    """Стоимость брони учитывает число номеров; флаги агента, компании и бесплатности."""
    row = build_features(booking).iloc[0]
    assert row["booking_value"] == pytest.approx(100.0 * 3 * 2)
    assert (row["has_agent"], row["has_company"], row["is_free"]) == (1, 0, 0)


def test_weekend_share_without_nights(booking):
    """Без ночей доля выходных равна 0, а не ошибке деления."""
    zero = booking.assign(stays_in_weekend_nights=0, stays_in_week_nights=0)
    assert build_features(zero).iloc[0]["weekend_share"] == 0.0


def test_columns(booking):
    """Исходные колонки даты удалены, колонки-утечки сохранены для EDA."""
    out = build_features(booking)
    assert not {"arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"} & set(out.columns)
    assert set(LEAKAGE_COLUMNS) <= set(out.columns)
