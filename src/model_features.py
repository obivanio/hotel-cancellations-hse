"""Признаки модели: что известно в момент брони, и что исключено с причиной."""
import pandas as pd

from src.features import LEAKAGE_COLUMNS

TARGET = "is_canceled"
NUMERIC = [
    "lead_time", "nights_weekend", "nights_week", "adults", "children", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled", "has_agent", "has_company",
    "days_in_waiting_list", "adr", "total_of_special_requests", "n_rooms", "arrival_month", "arrival_weekday",
    "is_free",
]
CATEGORICAL = [
    "hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type", "deposit_type", "customer_type",
]
FEATURES = NUMERIC + CATEGORICAL
LOG_COLUMNS = ["lead_time", "adr"]
EXCLUDED = {
    **{column: "утечка: известно только после заезда или отмены" for column in LEAKAGE_COLUMNS},
    "required_car_parking_spaces": "подозрение на утечку: у броней с парковкой нет ни одной отмены",
    "country": "подозрение на утечку: страну часто уточняют при заезде",
}


def model_frame(df: pd.DataFrame, extra: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.Series]:
    """Матрица признаков X (категории — строки) и целевая переменная y."""
    columns = FEATURES + list(extra)
    X = df[columns].copy()
    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].astype(str)
    return X, df[TARGET]


def categorical_columns(X: pd.DataFrame) -> list[str]:
    """Нечисловые колонки матрицы признаков."""
    return [column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])]
