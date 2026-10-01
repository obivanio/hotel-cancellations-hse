"""Тесты журнала очистки."""
import pandas as pd
import pytest

from src.cleaning_log import COLLAPSE, MODIFY, REMOVE, CleaningLog, LogEntry


def _entry(step, action, before, after):
    """Запись журнала с типовым обоснованием."""
    return LogEntry(step, action, f"правило {step}", before, after, before - after, "обоснование")


def test_add_rejects_empty_reason():
    """Шаг без обоснования в журнал не попадает."""
    log = CleaningLog()
    with pytest.raises(ValueError):
        log.add(LogEntry("1", REMOVE, "правило", 10, 9, 1, "   "))


def test_rows_removed_counts_only_removals():
    """Удалёнными считаются только шаги remove, склейка — нет."""
    log = CleaningLog()
    log.add(_entry("1", REMOVE, 100, 90))
    log.add(_entry("7", MODIFY, 90, 90))
    log.add(_entry("12", COLLAPSE, 90, 60))
    assert log.rows_removed() == 10


def test_is_continuous():
    """Журнал непрерывен, если шаги стыкуются по числу строк."""
    log = CleaningLog()
    log.add(_entry("1", REMOVE, 100, 90))
    log.add(_entry("12", COLLAPSE, 90, 60))
    assert log.is_continuous(100, 60)
    assert not log.is_continuous(101, 60)
    assert not log.is_continuous(100, 59)
    broken = CleaningLog()
    broken.add(_entry("1", REMOVE, 100, 90))
    broken.add(_entry("2", REMOVE, 80, 70))
    assert not broken.is_continuous(100, 70)


def test_to_frame_and_save(tmp_path):
    """Журнал превращается в таблицу и сохраняется в CSV без потерь."""
    log = CleaningLog()
    log.add(_entry("1", REMOVE, 100, 90))
    frame = log.to_frame()
    assert list(frame.columns) == [
        "step", "action", "rule", "rows_before", "rows_after", "rows_affected", "reason",
    ]
    path = tmp_path / "reports" / "log.csv"
    log.save(path)
    loaded = pd.read_csv(path, dtype={"step": str})
    pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)
