"""Тесты правил удаления (1–6) и приведения типа children (10)."""
import numpy as np
import pytest

from src.cleaning import REMOVAL_RULES, apply_removal, fix_children_type
from src.cleaning_log import MODIFY, REMOVE, CleaningLog


def _rule(step):
    """Правило удаления по номеру шага."""
    return next(rule for rule in REMOVAL_RULES if rule.step == step)


@pytest.mark.parametrize(
    ("step", "bad_row"),
    [
        ("1", {"adr": -6.38}),
        ("2", {"adr": 5400.0}),
        ("3", {"children": 10.0}),
        ("3", {"babies": 9}),
        ("3", {"adults": 55}),
        ("4", {"stays_in_weekend_nights": 0, "stays_in_week_nights": 0}),
        ("5", {"adults": 0, "children": 0.0, "babies": 0}),
        ("6", {"market_segment": "Undefined"}),
        ("6", {"distribution_channel": "Undefined"}),
    ],
)
def test_removal_rule_drops_only_bad_row(make_raw, step, bad_row):
    """Правило удаляет ровно ошибочную строку и пишет это в журнал с обоснованием."""
    df = make_raw([bad_row, {}])
    log = CleaningLog()
    out = apply_removal(df, _rule(step), log)
    assert len(out) == 1
    entry = log.entries[0]
    assert (entry.step, entry.action, entry.rows_before, entry.rows_after, entry.rows_affected) == (
        step, REMOVE, 2, 1, 1,
    )
    assert entry.reason.strip()


@pytest.mark.parametrize(
    ("step", "edge_row"),
    [
        ("2", {"adr": 510.0}),
        ("3", {"adults": 10, "children": 8.0, "babies": 8}),
        ("4", {"stays_in_weekend_nights": 0, "stays_in_week_nights": 1}),
        ("5", {"adults": 0, "children": 1.0, "babies": 0}),
    ],
)
def test_removal_rule_keeps_boundary_rows(make_raw, step, edge_row):
    """Граничные, но допустимые значения не удаляются."""
    df = make_raw([edge_row])
    out = apply_removal(df, _rule(step), CleaningLog())
    assert len(out) == 1


def test_removal_rules_order():
    """Шаги удаления идут по порядку 1–6."""
    assert [rule.step for rule in REMOVAL_RULES] == ["1", "2", "3", "4", "5", "6"]


def test_fix_children_type(make_raw):
    """Пропуски в children → 0, тип — целое; в журнале число исправленных строк."""
    df = make_raw([{"children": np.nan}, {"children": 1.0}])
    log = CleaningLog()
    out = fix_children_type(df, log)
    assert out["children"].tolist() == [0, 1]
    assert out["children"].dtype.kind == "i"
    entry = log.entries[0]
    assert (entry.step, entry.action, entry.rows_affected) == ("10", MODIFY, 1)
