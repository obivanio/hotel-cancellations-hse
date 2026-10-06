"""Сплит броней на обучение, валидацию и тест: по времени заезда (основной) и случайный (для сравнения)."""
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_SHARE = 0.6
VALID_SHARE = 0.2


def time_split(df: pd.DataFrame, train_share: float = TRAIN_SHARE,
               valid_share: float = VALID_SHARE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделить брони по дате заезда: ранние — обучение, следующие — валидация, поздние — тест.

    Доли частей — train_share и valid_share (по умолчанию 60 % и 20 %), тест — остаток.
    Брони с одной датой заезда всегда попадают в одну часть.
    """
    dates = df["arrival_date"]
    valid_start = dates.quantile(train_share)
    test_start = dates.quantile(train_share + valid_share)
    return df[dates < valid_start], df[(dates >= valid_start) & (dates < test_start)], df[dates >= test_start]


def random_split(df: pd.DataFrame, random_state: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Случайно разделить брони в тех же долях (60/20/20 %) с одинаковой долей отмен во всех частях.

    Только для сравнения со сплитом по времени: так модель видит брони тех же месяцев, что и в тесте.
    """
    rest, test = train_test_split(df, test_size=1 - TRAIN_SHARE - VALID_SHARE, stratify=df["is_canceled"],
                                  random_state=random_state)
    train, valid = train_test_split(rest, test_size=VALID_SHARE / (TRAIN_SHARE + VALID_SHARE),
                                    stratify=rest["is_canceled"], random_state=random_state)
    return train, valid, test


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
