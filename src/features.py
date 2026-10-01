"""Признаки броней для анализа и модели."""
import pandas as pd

LEAKAGE_COLUMNS = ("reservation_status", "reservation_status_date", "assigned_room_type", "booking_changes")
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}
SEASONS = {
    12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn",
}
RAW_DATE_COLUMNS = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить даты, параметры проживания, деньги и календарь; убрать исходные колонки даты."""
    out = df.rename(columns={"stays_in_week_nights": "nights_week", "stays_in_weekend_nights": "nights_weekend"})
    out["arrival_date"] = pd.to_datetime(pd.DataFrame({
        "year": out["arrival_date_year"],
        "month": out["arrival_date_month"].map(MONTHS),
        "day": out["arrival_date_day_of_month"],
    }))
    out["booking_date"] = out["arrival_date"] - pd.to_timedelta(out["lead_time"], unit="D")
    out["reservation_status_date"] = pd.to_datetime(out["reservation_status_date"])
    out["nights_total"] = out["nights_week"] + out["nights_weekend"]
    out["weekend_share"] = (out["nights_weekend"] / out["nights_total"].where(out["nights_total"] > 0)).fillna(0.0)
    out["guests"] = out["adults"] + out["children"] + out["babies"]
    out["is_family"] = ((out["children"] + out["babies"]) > 0).astype(int)
    out["has_agent"] = out["agent"].notna().astype(int)
    out["has_company"] = out["company"].notna().astype(int)
    out["is_free"] = (out["adr"] == 0).astype(int)
    out["booking_value"] = out["adr"] * out["nights_total"] * out["n_rooms"]
    out["arrival_month"] = out["arrival_date"].dt.month
    out["arrival_weekday"] = out["arrival_date"].dt.dayofweek
    out["season"] = out["arrival_month"].map(SEASONS)
    return out.drop(columns=RAW_DATE_COLUMNS)
