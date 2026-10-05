"""Тесты сплита по времени."""
import pandas as pd

from src.split import split_summary, time_split


def test_time_split_order_and_coverage():
    """Части не пересекаются, покрывают все брони и идут строго друг за другом по дате заезда."""
    dates = pd.date_range("2016-01-01", periods=100, freq="D").repeat(2)
    df = pd.DataFrame({"arrival_date": dates, "booking_key": range(200)})
    train, valid, test = time_split(df)
    assert len(train) + len(valid) + len(test) == len(df)
    assert set(train.index).isdisjoint(valid.index) and set(valid.index).isdisjoint(test.index)
    assert train["arrival_date"].max() < valid["arrival_date"].min()
    assert valid["arrival_date"].max() < test["arrival_date"].min()
    assert abs(len(train) / len(df) - 0.6) < 0.05


def test_time_split_same_date_same_part():
    """Брони с одной датой заезда не разрываются между частями."""
    dates = pd.to_datetime(["2016-01-01"] * 5 + ["2016-02-01"] * 5 + ["2016-03-01"] * 5)
    train, valid, test = time_split(pd.DataFrame({"arrival_date": dates}))
    for part in (train, valid, test):
        assert part["arrival_date"].nunique() <= 1


def test_split_summary():
    """Сводка сплита: число броней, первая и последняя дата заезда, доля отмен."""
    part = pd.DataFrame({"arrival_date": pd.to_datetime(["2016-01-01", "2016-01-03"]), "is_canceled": [0, 1]})
    summary = split_summary({"train": part})
    assert summary.loc["train", "bookings"] == 2
    assert str(summary.loc["train", "last_arrival"]) == "2016-01-03"
    assert summary.loc["train", "cancel_rate"] == 0.5
