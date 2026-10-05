"""Сплит броней по времени заезда: обучение, валидация, тест."""
import pandas as pd

TRAIN_SHARE = 0.6
VALID_SHARE = 0.2


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделить брони по дате заезда: ранние 60 % — обучение, следующие 20 % — валидация, поздние — тест.

    Брони с одной датой заезда всегда попадают в одну часть.
    """
    dates = df["arrival_date"]
    valid_start = dates.quantile(TRAIN_SHARE)
    test_start = dates.quantile(TRAIN_SHARE + VALID_SHARE)
    return df[dates < valid_start], df[(dates >= valid_start) & (dates < test_start)], df[dates >= test_start]


def split_summary(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Размер, границы дат заезда и доля отмен каждой части сплита."""
    return pd.DataFrame({
        name: {
            "bookings": len(part),
            "first_arrival": part["arrival_date"].min().date(),
            "last_arrival": part["arrival_date"].max().date(),
            "cancel_rate": round(float(part["is_canceled"].mean()), 3),
        } for name, part in parts.items()
    }).T
