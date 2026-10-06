"""Интеграционные тесты шага 4: правила и овербукинг на реальных данных и сохранённой модели."""
import numpy as np
import pytest

from src.data import load_raw
from src.model_features import model_frame
from src.models import fit_calibrator, load_catboost
from src.overbooking import benchmark_proba, benchmark_rates, date_table
from src.pipeline import build_dataset
from src.policies import ALL_HOTELS, best_lead_days, policy_table, scored
from src.split import time_split


@pytest.fixture(name="real", scope="module")
def fixture_real():
    """Обучение, валидация, тест и калиброванные вероятности сохранённой модели (как в ноутбуке 04)."""
    df, _ = build_dataset(load_raw())
    train, valid, test = time_split(df)
    model = load_catboost()
    (X_valid, y_valid), (X_test, _) = model_frame(valid), model_frame(test)
    raw_valid = model.predict_proba(X_valid)[:, 1]
    calibrator = fit_calibrator(raw_valid, y_valid)
    return {"train": train, "valid": valid, "test": test, "valid_proba": calibrator.predict(raw_valid),
            "test_proba": calibrator.predict(model.predict_proba(X_test)[:, 1])}


def test_real_data_policies(real):
    """Правила на тесте: доли в [0, 1], потери неотрицательны, топ-30 % — ровно 30 % броней в каждой части."""
    lead_days = best_lead_days(scored(real["valid"], real["valid_proba"]))
    table = policy_table(scored(real["test"], real["test_proba"]), lead_days, threshold=0.17)
    assert table[["prepay_share", "touched_reliable", "caught_cancels"]].stack().between(0, 1).all()
    assert (table["total"] >= 0).all()
    top = table[table["policy"] == "модель, топ-30 %"]
    assert top["prepay_share"].to_numpy() == pytest.approx(np.full(3, 0.3), abs=1e-4)


def test_real_data_overbooking(real):
    """Овербукинг на тесте: среднее ожидание отмен на дату близко к факту, осторожная оценка не больше броней."""
    train, test = real["train"], real["test"]
    bench = benchmark_proba(benchmark_rates(train), test, fallback=train["is_canceled"].mean())
    dates = date_table(test, real["test_proba"], bench)
    pooled = dates[dates["hotel"] == ALL_HOTELS]
    assert pooled["expected"].mean() == pytest.approx(pooled["actual"].mean(), rel=0.1)
    assert (dates["cautious"] <= dates["bookings"]).all() and dates["bench_expected"].notna().all()
