"""Тесты обучения моделей на синтетических бронях."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.model_features import CATEGORICAL, FEATURES, model_frame
from src.models import (
    as_category, fit_calibrator, fit_catboost, fit_hist_gb, fit_logistic, load_catboost, odds_ratios, save_catboost,
)


def _frame(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Синтетические брони: отмена тем вероятнее, чем больше срок брони и если сегмент Online TA."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({column: rng.integers(0, 3, n) for column in FEATURES})
    for column in CATEGORICAL:
        df[column] = rng.choice(["a", "b"], n)
    df["market_segment"] = rng.choice(["Online TA", "Direct"], n)
    df["lead_time"] = rng.integers(0, 400, n)
    df["adr"] = rng.uniform(0, 200, n)
    prob = 0.1 + 0.6 * df["lead_time"] / 400 + 0.2 * (df["market_segment"] == "Online TA")
    df["is_canceled"] = (rng.random(n) < prob).astype(int)
    return model_frame(df)


@pytest.fixture(name="data", scope="module")
def fixture_data():
    """Обучающая и проверочная синтетика."""
    return _frame(1500, 0), _frame(500, 1)


def test_models_beat_random(data):
    """Каждая модель на синтетике с эффектом лучше случайного угадывания."""
    (X, y), (X_new, y_new) = data
    assert roc_auc_score(y_new, fit_logistic(X, y).predict_proba(X_new)[:, 1]) > 0.6
    assert roc_auc_score(y_new, fit_hist_gb(X, y).predict_proba(as_category(X_new))[:, 1]) > 0.6
    assert roc_auc_score(y_new, fit_catboost((X, y), (X_new, y_new)).predict_proba(X_new)[:, 1]) > 0.6


def test_catboost_save_load(data, tmp_path):
    """Сохранённая и загруженная модель даёт те же прогнозы."""
    (X, y), (X_new, y_new) = data
    model = fit_catboost((X, y), (X_new, y_new))
    loaded = load_catboost(save_catboost(model, tmp_path / "model.cbm"))
    assert np.allclose(model.predict_proba(X_new), loaded.predict_proba(X_new))


def test_calibrator_monotonic():
    """Калибратор переводит сырые вероятности в доли отмен и не ломает порядок."""
    proba = np.linspace(0, 1, 200)
    y = pd.Series((proba > 0.5).astype(int))
    calibrated = fit_calibrator(proba, y).predict(proba)
    assert np.all(np.diff(calibrated) >= 0)
    assert calibrated[0] == pytest.approx(0.0) and calibrated[-1] == pytest.approx(1.0)


def test_odds_ratios(data):
    """Отношения шансов положительны и названы по признакам; сильнейшие — первыми."""
    (X, y), _ = data
    odds = odds_ratios(fit_logistic(X, y))
    assert (odds > 0).all()
    assert odds.index[0].startswith(("log__lead_time", "cat__market_segment"))
    assert abs(np.log(odds.iloc[0])) >= abs(np.log(odds.iloc[-1]))
