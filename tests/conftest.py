"""Общие фикстуры: маленькие таблицы в формате исходного датасета."""
import numpy as np
import pandas as pd
import pytest

BASE_ROW = {
    "hotel": "City Hotel",
    "is_canceled": 0,
    "lead_time": 30,
    "arrival_date_year": 2016,
    "arrival_date_month": "July",
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 1,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0.0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "agent": 9.0,
    "company": np.nan,
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 100.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1,
    "reservation_status": "Check-Out",
    "reservation_status_date": "2016-07-04",
}


@pytest.fixture(name="make_raw")
def fixture_make_raw():
    """Фабрика таблиц: каждая строка — BASE_ROW с переопределёнными полями."""

    def _make(rows):
        return pd.DataFrame([{**BASE_ROW, **row} for row in rows])

    return _make
