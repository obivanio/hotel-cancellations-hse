"""Тесты правил-бенчмарков."""
import pandas as pd
import pytest

from src.baselines import lead_segment_rates, lead_segment_score, lead_time_score


def test_lead_time_score():
    """Скор правила «срок» — сам срок брони."""
    assert lead_time_score(pd.DataFrame({"lead_time": [3, 400]})).tolist() == [3.0, 400.0]


def test_lead_segment_rule():
    """Доли отмен по ячейкам «срок × сегмент» на обучении; неизвестная ячейка — общая доля."""
    train = pd.DataFrame({
        "lead_time": [1, 2, 100, 100],
        "market_segment": ["Direct", "Direct", "Groups", "Groups"],
        "is_canceled": [0, 1, 1, 1],
    })
    rates = lead_segment_rates(train)
    new = pd.DataFrame({"lead_time": [5, 120, 5], "market_segment": ["Direct", "Groups", "Groups"]})
    assert lead_segment_score(rates, new, fallback=0.75).tolist() == pytest.approx([0.5, 1.0, 0.75])
