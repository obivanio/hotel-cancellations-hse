"""Сборка очищенного датасета, проверка гарантий, сравнение вариантов и сохранение результатов."""
from pathlib import Path

import pandas as pd

from src.cleaning import ADR_ERROR_THRESHOLD, VARIANTS, make_variant
from src.cleaning_log import CleaningLog
from src.features import build_features

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "bookings_clean.parquet"
LOG_PATH = ROOT / "reports" / "cleaning_log.csv"
NA_ALLOWED = {"agent", "agent_id", "company"}
CATEGORIES_WITHOUT_UNDEFINED = ["meal", "market_segment", "distribution_channel"]


def build_dataset(df_raw: pd.DataFrame, variant: str = "C") -> tuple[pd.DataFrame, CleaningLog]:
    """Очистить сырые данные по варианту обработки дублей и добавить признаки."""
    df, log = make_variant(df_raw, variant)
    return build_features(df), log


def check_guarantees(df: pd.DataFrame, n_raw: int, log: CleaningLog) -> dict[str, bool]:
    """Проверить 6 гарантий очищенного датасета (§5 спеки подпроекта 1)."""
    content = df.drop(columns=["booking_key", "n_rooms"])
    na_columns = set(df.columns[df.isna().any()])
    return {
        "1. нет одинаковых строк, n_rooms ≥ 1": bool(
            not content.duplicated().any() and (df["n_rooms"] >= 1).all()
        ),
        "2. сумма n_rooms = строк после удалений": int(df["n_rooms"].sum()) == n_raw - log.rows_removed(),
        "3. ночей ≥ 1, гостей ≥ 1, 0 ≤ adr < порога": bool(
            (df["nights_total"] >= 1).all()
            and (df["guests"] >= 1).all()
            and df["adr"].between(0, ADR_ERROR_THRESHOLD, inclusive="left").all()
        ),
        "4. нет Undefined в категориях": not bool(df[CATEGORIES_WITHOUT_UNDEFINED].eq("Undefined").any().any()),
        "5. пропуски только в agent, agent_id и company": na_columns <= NA_ALLOWED,
        "6. журнал непрерывен": log.is_continuous(n_raw, len(df)),
    }


def compare_variants(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Доля отмен по броням и номерам и доля отменённой выручки для вариантов A, B, C."""
    rows = []
    for variant in VARIANTS:
        df, _ = build_dataset(df_raw, variant)
        rooms = df["n_rooms"]
        canceled = df["is_canceled"] == 1
        rows.append({
            "variant": variant,
            "bookings": len(df),
            "rooms": int(rooms.sum()),
            "cancel_rate_bookings": float(canceled.mean()),
            "cancel_rate_rooms": float(rooms[canceled].sum() / rooms.sum()),
            "canceled_value_share": float(df.loc[canceled, "booking_value"].sum() / df["booking_value"].sum()),
        })
    return pd.DataFrame(rows).set_index("variant")


def save_outputs(
    df: pd.DataFrame, log: CleaningLog, clean_path: Path = CLEAN_PATH, log_path: Path = LOG_PATH
) -> None:
    """Сохранить очищенный датасет (parquet) и журнал очистки (CSV)."""
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(clean_path, index=False)
    log.save(log_path)
