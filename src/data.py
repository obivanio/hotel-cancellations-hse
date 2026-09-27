"""Загрузка исходного датасета Hotel Booking Demand."""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-02-11/hotels.csv"
)
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "hotels.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Вернуть исходную таблицу броней, при первом вызове скачав её в data/raw."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, path)
    return pd.read_csv(path)
