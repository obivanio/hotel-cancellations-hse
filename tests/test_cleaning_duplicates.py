"""Тесты склейки дублей, редких агентов, вариантов A/B/C и полной очистки."""
import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean, collapse_duplicates, group_rare_agents, make_variant
from src.cleaning_log import COLLAPSE, CleaningLog


def test_collapse_merges_identical_rows(make_raw):
    """Три одинаковые строки → одна бронь с n_rooms = 3; отличающаяся строка отдельно."""
    df = make_raw([{}, {}, {}, {"adr": 120.0}])
    log = CleaningLog()
    out = collapse_duplicates(df, log)
    assert len(out) == 2
    assert sorted(out["n_rooms"].tolist()) == [1, 3]
    assert out["n_rooms"].sum() == 4
    assert out["booking_key"].tolist() == [0, 1]
    assert out.columns[0] == "booking_key"
    assert out.columns[-1] == "n_rooms"
    entry = log.entries[0]
    assert (entry.step, entry.action, entry.rows_before, entry.rows_after, entry.rows_affected) == (
        "12", COLLAPSE, 4, 2, 2,
    )


def test_collapse_merges_rows_with_missing_values(make_raw):
    """Пропуски (agent, company) не мешают склейке одинаковых строк."""
    df = make_raw([{"agent": np.nan}, {"agent": np.nan}])
    out = collapse_duplicates(df, CleaningLog())
    assert out["n_rooms"].tolist() == [2]


def test_collapse_keeps_rows_differing_in_one_field(make_raw):
    """Строки, отличающиеся хотя бы одним полем, не склеиваются."""
    df = make_raw([{"lead_time": 30}, {"lead_time": 31}])
    out = collapse_duplicates(df, CleaningLog())
    assert out["n_rooms"].tolist() == [1, 1]


def test_group_rare_agents(make_raw):
    """Агенты с числом броней меньше порога → other; пусто остаётся пустым."""
    df = make_raw([{"agent": 9.0}, {"agent": 9.0}, {"agent": 240.0}, {"agent": np.nan}])
    log = CleaningLog()
    out = group_rare_agents(df, log, min_bookings=2)
    assert out["agent"].tolist()[:3] == ["9", "9", "other"]
    assert out["agent_id"].tolist()[:3] == ["9", "9", "240"]
    assert pd.isna(out["agent"].iloc[3]) and pd.isna(out["agent_id"].iloc[3])
    assert (log.entries[0].step, log.entries[0].rows_affected) == ("9", 1)


@pytest.mark.parametrize(("variant", "rows", "rooms"), [("A", 2, 2), ("B", 4, 4), ("C", 2, 4)])
def test_make_variant(make_raw, variant, rows, rooms):
    """A — дубли удалены, B — оставлены, C — склеены с сохранением числа номеров."""
    df_raw = make_raw([{}, {}, {}, {"adr": 120.0}])
    out, log = make_variant(df_raw, variant)
    assert len(out) == rows
    assert out["n_rooms"].sum() == rooms
    assert log.is_continuous(len(df_raw), len(out))


def test_make_variant_rejects_unknown():
    """Неизвестный вариант — ошибка."""
    with pytest.raises(ValueError):
        make_variant(pd.DataFrame(), "D")


def test_clean_is_variant_c(make_raw):
    """clean — это вариант C: ошибочные строки удалены, дубли склеены."""
    df_raw = make_raw([{}, {}, {"adr": -1.0}, {"stays_in_weekend_nights": 0, "stays_in_week_nights": 0}])
    out, log = clean(df_raw)
    assert len(out) == 1
    assert out["n_rooms"].tolist() == [2]
    assert log.rows_removed() == 2
    assert out["n_rooms"].sum() == len(df_raw) - log.rows_removed()
