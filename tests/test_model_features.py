"""Тесты набора признаков модели."""
import pandas as pd

from src.features import LEAKAGE_COLUMNS
from src.model_features import CATEGORICAL, EXCLUDED, FEATURES, model_frame


def test_no_excluded_in_features():
    """Ни утечки, ни подозрительные колонки не входят в признаки; у каждого исключения есть причина."""
    assert set(LEAKAGE_COLUMNS) <= set(EXCLUDED)
    assert {"required_car_parking_spaces", "country"} <= set(EXCLUDED)
    assert not set(EXCLUDED) & set(FEATURES)
    assert all(reason for reason in EXCLUDED.values())


def test_model_frame_types():
    """Категории приводятся к строкам, числовые остаются числами; y — целевая колонка."""
    row = {column: 1 for column in FEATURES} | {column: "x" for column in CATEGORICAL} | {"is_canceled": 1}
    X, y = model_frame(pd.DataFrame([row, row]))
    assert list(X.columns) == FEATURES
    assert X["hotel"].tolist() == ["x", "x"]
    assert pd.api.types.is_numeric_dtype(X["lead_time"])
    assert y.tolist() == [1, 1]
