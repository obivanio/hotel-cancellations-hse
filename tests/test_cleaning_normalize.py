"""Тесты нормализации: питание Undefined, пустая страна, номер недели."""
import numpy as np

from src.cleaning import drop_week_number, fill_unknown_country, merge_undefined_meal
from src.cleaning_log import MODIFY, CleaningLog


def test_merge_undefined_meal(make_raw):
    """Питание Undefined объединяется с SC (без пакета питания)."""
    df = make_raw([{"meal": "Undefined"}, {"meal": "SC"}, {"meal": "BB"}])
    log = CleaningLog()
    out = merge_undefined_meal(df, log)
    assert out["meal"].tolist() == ["SC", "SC", "BB"]
    entry = log.entries[0]
    assert (entry.step, entry.action, entry.rows_affected, entry.rows_after) == ("7", MODIFY, 1, 3)


def test_fill_unknown_country(make_raw):
    """Пустая страна становится отдельной категорией unknown."""
    df = make_raw([{"country": np.nan}, {}])
    log = CleaningLog()
    out = fill_unknown_country(df, log)
    assert out["country"].tolist() == ["unknown", "PRT"]
    assert (log.entries[0].step, log.entries[0].rows_affected) == ("8", 1)


def test_drop_week_number(make_raw):
    """Колонка номера недели удаляется, строки не меняются."""
    df = make_raw([{}, {}])
    log = CleaningLog()
    out = drop_week_number(df, log)
    assert "arrival_date_week_number" not in out.columns
    assert len(out) == 2
    assert (log.entries[0].step, log.entries[0].rows_affected) == ("11", 2)
