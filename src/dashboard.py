"""Эталонные расчёты дашборда: срез броней по фильтрам, порог модели, подписи чисел.

Браузер (src/dashboard_assets/dashboard.js) повторяет эти формулы, тест сверки проверяет совпадение.
"""
import numpy as np
import pandas as pd

from src.evaluation import DEFAULT_ASSUMPTIONS, Assumptions, value_at_risk
from src.policies import ALL_HOTELS, REPORTS_DIR, Scored

SCORED_TEST_PATH = REPORTS_DIR / "scored_test.csv"
SCORED_COLUMNS = ["hotel", "market_segment", "arrival_month", "lead_time", "value", "is_canceled", "proba"]
SMALL_MONEY = 100_000


def scored_frame(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Таблица для дашборда: бронь → отель, сегмент, месяц, срок, стоимость при отмене, факт отмены, риск модели.

    value — стоимость, которую отель теряет при отмене (у броней «без возврата» — 0), как в правилах шага 4.
    """
    return pd.DataFrame({
        "hotel": df["hotel"].astype(str).to_numpy(), "market_segment": df["market_segment"].astype(str).to_numpy(),
        "arrival_month": df["arrival_month"].to_numpy(dtype=int), "lead_time": df["lead_time"].to_numpy(dtype=float),
        "value": value_at_risk(df), "is_canceled": df["is_canceled"].to_numpy(dtype=int),
        "proba": np.asarray(proba, dtype=float),
    })


def filter_frame(frame: pd.DataFrame, hotel: str, segments: list[str], months: list[int]) -> pd.DataFrame:
    """Брони выбранного отеля («все отели» — оба), выбранных сегментов и месяцев заезда."""
    mask = frame["market_segment"].isin(segments) & frame["arrival_month"].isin(months)
    if hotel != ALL_HOTELS:
        mask &= frame["hotel"] == hotel
    return frame[mask]


def to_scored(frame: pd.DataFrame) -> Scored:
    """Scored из таблицы scored_frame — вход функций src/policies.py."""
    return Scored(canceled=frame["is_canceled"].to_numpy() == 1, proba=frame["proba"].to_numpy(dtype=float),
                  lead_time=frame["lead_time"].to_numpy(dtype=float), value=frame["value"].to_numpy(dtype=float),
                  hotel=frame["hotel"].to_numpy(dtype=str))


def model_threshold(assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> float:
    """Порог модели loss_prob / (unsold_share + loss_prob): брать предоплату, когда ожидаемая потеря от отмены
    больше ожидаемой потери от ушедшего гостя (для калиброванных вероятностей)."""
    return assumptions.loss_prob / (assumptions.unsold_share + assumptions.loss_prob)


def comma(value: float, digits: int = 2) -> str:
    """Число с десятичной запятой: 0,17 (без «-0,00» у малых отрицательных)."""
    return f"{round(value, digits) + 0.0:.{digits}f}".replace(".", ",")


def money(value: float) -> str:
    """Сумма с десятичной запятой: от 0,1 млн € — в миллионах (2,11 млн €), меньше — в тысячах (23,4 тыс. €)."""
    if abs(value) < SMALL_MONEY:
        return f"{comma(value / 1e3, 1)} тыс. €"
    return f"{comma(value / 1e6)} млн €"


def count(value: float) -> str:
    """Целое число с пробелом между тысячами: 4 898."""
    return f"{value:,.0f}".replace(",", " ")


def percent(share: float) -> str:
    """Доля в процентах без знаков после запятой: 60 %."""
    return f"{share * 100:.0f} %"
