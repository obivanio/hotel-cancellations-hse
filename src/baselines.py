"""Правила-бенчмарки: срок брони и «срок + сегмент»."""
import numpy as np
import pandas as pd

from src.eda import lead_time_bins


def lead_time_score(df: pd.DataFrame) -> np.ndarray:
    """Правило «чем раньше бронь, тем выше риск»: скор = срок брони в днях."""
    return df["lead_time"].to_numpy(dtype=float)


def lead_segment_rates(train: pd.DataFrame) -> pd.Series:
    """Доля отмен в каждой ячейке «корзина срока брони × сегмент» на обучающих данных."""
    keys = [lead_time_bins(train["lead_time"]).astype(str), train["market_segment"].astype(str)]
    return train.groupby(keys)["is_canceled"].mean()


def lead_segment_score(rates: pd.Series, df: pd.DataFrame, fallback: float) -> np.ndarray:
    """Правило «срок + сегмент»: скор = доля отмен ячейки на обучении; неизвестная ячейка — общая доля."""
    keys = pd.MultiIndex.from_arrays([lead_time_bins(df["lead_time"]).astype(str), df["market_segment"].astype(str)])
    return rates.reindex(keys).fillna(fallback).to_numpy(dtype=float)
