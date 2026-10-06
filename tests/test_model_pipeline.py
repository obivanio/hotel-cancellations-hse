"""Интеграционный тест: полный цикл на реальных данных."""
from src.data import load_raw
from src.evaluation import best_threshold, loss_curve, value_at_risk
from src.model_features import model_frame
from src.models import fit_catboost
from src.pipeline import build_dataset
from src.split import time_split


def test_real_data_cycle():
    """Сплит → CatBoost → порог по деньгам строятся на реальных данных; порог в [0, 1]."""
    df, _ = build_dataset(load_raw())
    train, valid, _ = time_split(df)
    X_valid, y_valid = model_frame(valid)
    model = fit_catboost(model_frame(train), (X_valid, y_valid))
    curve = loss_curve(y_valid, model.predict_proba(X_valid)[:, 1], value_at_risk(valid))
    assert 0.0 <= best_threshold(curve) <= 1.0
