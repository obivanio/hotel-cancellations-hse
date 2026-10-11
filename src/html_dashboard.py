"""Сборка дашборда reports/dashboard.html: один файл с данными, оформлением, логикой и шрифтами, без сети."""
import numpy as np
import pandas as pd

from src.pipeline import ROOT

ASSETS_DIR = ROOT / "src" / "dashboard_assets"


def pack_bookings(frame: pd.DataFrame) -> dict:
    """Брони столбцами без повторения имён полей: отель и сегмент — номера в списках, риск — номер в списке значений.

    Порядок строк сохраняется: от него зависит порядок броней с одинаковым риском в очереди.
    """
    hotels = sorted(frame["hotel"].unique())
    segments = sorted(frame["market_segment"].unique())
    probas, proba_index = np.unique(frame["proba"].to_numpy(dtype=float), return_inverse=True)
    return {
        "hotels": hotels, "segments": segments,
        "h": frame["hotel"].map(hotels.index).tolist(), "s": frame["market_segment"].map(segments.index).tolist(),
        "m": frame["arrival_month"].astype(int).tolist(), "lead": frame["lead_time"].astype(float).tolist(),
        "value": frame["value"].astype(float).tolist(), "cancel": frame["is_canceled"].astype(int).tolist(),
        "probas": probas.tolist(), "p": proba_index.astype(int).tolist(),
    }
