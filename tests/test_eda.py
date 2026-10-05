"""Тесты таблиц разведочного анализа."""
import pandas as pd
import pytest

from src.eda import cancel_rates, lead_time_bins, lost_value_by_month, median_price_by_month


def _bookings():
    """Четыре брони: групповая отменённая бронь на 3 номера, бесплатная бронь, разные сроки."""
    return pd.DataFrame({
        "hotel": ["City Hotel", "City Hotel", "Resort Hotel", "Resort Hotel"],
        "is_canceled": [1, 0, 1, 0],
        "n_rooms": [3, 1, 1, 1],
        "booking_value": [600.0, 200.0, 100.0, 100.0],
        "arrival_date": pd.to_datetime(["2016-07-01", "2016-07-15", "2016-08-01", "2016-08-02"]),
        "arrival_month": [7, 7, 8, 8],
        "adr": [100.0, 120.0, 0.0, 80.0],
        "is_free": [0, 0, 1, 0],
        "lead_time": [0, 7, 8, 400],
    })


def test_lead_time_bins_boundaries():
    """Границы корзин: 7 — ещё «до 7 дней», 8 — уже «8–30 дней», 366 — «больше года»."""
    bins = lead_time_bins(pd.Series([0, 7, 8, 30, 31, 365, 366]))
    assert bins.astype(str).tolist() == [
        "до 7 дней", "до 7 дней", "8–30 дней", "8–30 дней", "31–90 дней", "181–365 дней", "больше года",
    ]


def test_cancel_rates_by_column():
    """Доля отмен по броням и по номерам: групповая бронь весит по числу номеров."""
    rates = cancel_rates(_bookings(), "hotel")
    assert rates.loc["City Hotel", "bookings"] == 2
    assert rates.loc["City Hotel", "cancel_rate"] == pytest.approx(0.5)
    assert rates.loc["City Hotel", "rooms"] == 4
    assert rates.loc["City Hotel", "cancel_rate_rooms"] == pytest.approx(0.75)
    assert rates.loc["Resort Hotel", "cancel_rate_rooms"] == pytest.approx(0.5)


def test_cancel_rates_by_series_drops_empty_bins():
    """Группировка по корзинам срока: пустые корзины не попадают в таблицу."""
    df = _bookings()
    rates = cancel_rates(df, lead_time_bins(df["lead_time"]))
    assert rates.index.astype(str).tolist() == ["до 7 дней", "8–30 дней", "больше года"]
    assert rates["bookings"].tolist() == [2, 1, 1]
    assert not rates.isna().any().any()


def test_lost_value_by_month():
    """Стоимость всех и отменённых броней по месяцам заезда и доля отменённой."""
    lost = lost_value_by_month(_bookings())
    assert lost["total_value"].tolist() == [800.0, 200.0]
    assert lost["canceled_value"].tolist() == [600.0, 100.0]
    assert lost["canceled_share"].tolist() == pytest.approx([0.75, 0.5])


def test_median_price_by_month_excludes_free():
    """Медианная цена по месяцам и отелям без бесплатных броней."""
    prices = median_price_by_month(_bookings())
    assert prices.loc[7, "City Hotel"] == pytest.approx(110.0)
    assert prices.loc[8, "Resort Hotel"] == pytest.approx(80.0)
