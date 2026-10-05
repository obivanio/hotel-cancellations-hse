"""Таблицы для разведочного анализа: доли отмен по группам, сроки брони, деньги по месяцам."""
import numpy as np
import pandas as pd

LEAD_TIME_BINS = [-1, 7, 30, 90, 180, 365, np.inf]
LEAD_TIME_LABELS = ["до 7 дней", "8–30 дней", "31–90 дней", "91–180 дней", "181–365 дней", "больше года"]


def lead_time_bins(lead_time: pd.Series) -> pd.Series:
    """Корзины срока брони (дней от бронирования до заезда)."""
    return pd.cut(lead_time, LEAD_TIME_BINS, labels=LEAD_TIME_LABELS)


def cancel_rates(df: pd.DataFrame, by: str | pd.Series) -> pd.DataFrame:
    """Доля отмен по группам: по броням-решениям и по номерам (с весом n_rooms)."""
    keys = df[by] if isinstance(by, str) else by
    frame = df.assign(_key=keys, _canceled_rooms=df["n_rooms"].where(df["is_canceled"] == 1, 0))
    grouped = frame.groupby("_key", observed=True)
    out = pd.DataFrame({
        "bookings": grouped.size(),
        "cancel_rate": grouped["is_canceled"].mean(),
        "rooms": grouped["n_rooms"].sum(),
        "cancel_rate_rooms": grouped["_canceled_rooms"].sum() / grouped["n_rooms"].sum(),
    })
    out.index.name = by if isinstance(by, str) else keys.name
    return out


def lost_value_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Стоимость всех и отменённых броней по месяцам заезда (с учётом n_rooms) и доля отменённой."""
    frame = df.assign(
        month=df["arrival_date"].dt.to_period("M"),
        canceled_value=df["booking_value"].where(df["is_canceled"] == 1, 0.0),
    )
    grouped = frame.groupby("month")
    out = pd.DataFrame({
        "total_value": grouped["booking_value"].sum(),
        "canceled_value": grouped["canceled_value"].sum(),
    })
    out["canceled_share"] = out["canceled_value"] / out["total_value"]
    return out


def median_price_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Медианная цена за ночь по месяцам заезда для каждого отеля, без бесплатных броней."""
    paid = df[df["is_free"] == 0]
    return paid.pivot_table(index="arrival_month", columns="hotel", values="adr", aggfunc="median")
