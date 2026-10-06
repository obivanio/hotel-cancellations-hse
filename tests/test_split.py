"""Тесты сплита по времени и случайного сплита."""
import pandas as pd

from src.split import random_split, split_summary, time_split


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


def test_time_split_custom_shares():
    """Доли можно задать: 70/15/15 даёт части нужных размеров, и время не перемешано."""
    dates = pd.date_range("2016-01-01", periods=200, freq="D").repeat(2)
    df = pd.DataFrame({"arrival_date": dates})
    train, valid, test = time_split(df, train_share=0.7, valid_share=0.15)
    assert abs(len(train) / len(df) - 0.7) < 0.02
    assert abs(len(valid) / len(df) - 0.15) < 0.02
    assert abs(len(test) / len(df) - 0.15) < 0.02
    assert train["arrival_date"].max() < valid["arrival_date"].min()
    assert valid["arrival_date"].max() < test["arrival_date"].min()


def _bookings(n: int = 1000) -> pd.DataFrame:
    """Брони с каждой четвёртой отменой (доля отмен 25 %)."""
    return pd.DataFrame({"booking_key": range(n), "is_canceled": [int(i % 4 == 0) for i in range(n)]})


def test_random_split_coverage_and_sizes():
    """Случайный сплит: части не пересекаются, покрывают все брони и имеют размеры около 60/20/20 %."""
    df = _bookings()
    train, valid, test = random_split(df)
    assert len(train) + len(valid) + len(test) == len(df)
    assert set(train.index).isdisjoint(valid.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(valid.index).isdisjoint(test.index)
    assert abs(len(train) / len(df) - 0.6) < 0.01
    assert abs(len(valid) / len(df) - 0.2) < 0.01
    assert abs(len(test) / len(df) - 0.2) < 0.01


def test_random_split_stratified_and_reproducible():
    """Доля отмен во всех частях почти одинаковая; при одном random_state результат тот же."""
    df = _bookings()
    parts = random_split(df, random_state=1)
    for part in parts:
        assert abs(part["is_canceled"].mean() - 0.25) < 0.01
    again = random_split(df, random_state=1)
    for first, second in zip(parts, again):
        assert first.index.equals(second.index)


def test_split_summary():
    """Сводка сплита: число броней, первая и последняя дата заезда, доля отмен."""
    part = pd.DataFrame({"arrival_date": pd.to_datetime(["2016-01-01", "2016-01-03"]), "is_canceled": [0, 1]})
    summary = split_summary({"train": part})
    assert summary.loc["train", "bookings"] == 2
    assert str(summary.loc["train", "last_arrival"]) == "2016-01-03"
    assert summary.loc["train", "cancel_rate"] == 0.5
