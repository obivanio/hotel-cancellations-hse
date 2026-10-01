"""Тесты сборки датасета, гарантий и сравнения вариантов; интеграционный тест на реальных данных."""
import pandas as pd
import pytest

from src.data import load_raw
from src.pipeline import build_dataset, check_guarantees, compare_variants, save_outputs


@pytest.fixture(name="group_raw")
def fixture_group_raw(make_raw):
    """Группа из трёх одинаковых отменённых номеров и одна неотменённая бронь."""
    return make_raw([{"is_canceled": 1}, {"is_canceled": 1}, {"is_canceled": 1}, {"adr": 120.0}])


def test_compare_variants(group_raw):
    """По броням варианты расходятся, по номерам C совпадает с B, A занижает."""
    table = compare_variants(group_raw)
    assert list(table.index) == ["A", "B", "C"]
    assert table.loc["B", "cancel_rate_bookings"] == pytest.approx(0.75)
    assert table.loc["C", "cancel_rate_bookings"] == pytest.approx(0.5)
    assert table.loc["C", "cancel_rate_rooms"] == pytest.approx(table.loc["B", "cancel_rate_rooms"])
    assert table.loc["A", "cancel_rate_rooms"] == pytest.approx(0.5)
    assert table.loc["C", "rooms"] == 4


def test_check_guarantees_on_small_table(group_raw):
    """На корректной таблице все гарантии выполнены."""
    df, log = build_dataset(group_raw)
    assert all(check_guarantees(df, len(group_raw), log).values())


def test_check_guarantees_variant_b_fails_duplicate_guarantee(group_raw):
    """Вариант B с одинаковыми строками не проходит гарантию 1 — в данных остаются полные дубли."""
    df, log = build_dataset(group_raw, "B")
    guarantees = check_guarantees(df, len(group_raw), log)
    assert guarantees["1. нет одинаковых строк, n_rooms ≥ 1"] is False


def test_save_outputs(tmp_path, group_raw):
    """Датасет и журнал сохраняются и читаются без потерь по размеру."""
    df, log = build_dataset(group_raw)
    clean_path, log_path = tmp_path / "clean.parquet", tmp_path / "log.csv"
    save_outputs(df, log, clean_path, log_path)
    assert pd.read_parquet(clean_path).shape == df.shape
    assert len(pd.read_csv(log_path)) == len(log.entries)


def test_real_data_guarantees():
    """Интеграционный: на реальных данных выполнены все 6 гарантий."""
    raw = load_raw()
    df, log = build_dataset(raw)
    failed = [name for name, ok in check_guarantees(df, len(raw), log).items() if not ok]
    assert not failed, failed


def test_real_data_variants_keep_rooms():
    """Интеграционный: склейка (C) сохраняет число номеров варианта B."""
    table = compare_variants(load_raw())
    assert table.loc["C", "rooms"] == table.loc["B", "rooms"]
    assert table.loc["C", "cancel_rate_rooms"] == pytest.approx(table.loc["B", "cancel_rate_rooms"])
