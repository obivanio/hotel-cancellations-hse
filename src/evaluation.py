"""Оценка моделей: метрики, калибровка, потери по порогу «брать предоплату» и чувствительность порога."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from src.pipeline import ROOT

METRICS_PATH = ROOT / "reports" / "model_metrics.csv"
THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)
NON_REFUND = "Non Refund"


@dataclass(frozen=True)
class Assumptions:
    """Допущения о деньгах (в данных их нет, проверяются анализом чувствительности).

    unsold_share — доля стоимости отменённой брони, которую не удаётся вернуть перепродажей номера;
    loss_prob — вероятность потерять надёжного гостя из-за требования предоплаты.
    """

    unsold_share: float = 0.5
    loss_prob: float = 0.1


DEFAULT_ASSUMPTIONS = Assumptions()


def score_table(y: pd.Series, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """PR-AUC и ROC-AUC каждой модели на одних и тех же данных."""
    return pd.DataFrame({
        name: {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p)} for name, p in scores.items()
    }).T


def calibration_table(y: pd.Series, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Средний прогноз и наблюдаемая доля отмен в корзинах прогноза (равные по числу броней)."""
    observed, predicted = calibration_curve(y, proba, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"predicted": predicted, "observed": observed})


def value_at_risk(df: pd.DataFrame) -> np.ndarray:
    """Стоимость, которую отель теряет при отмене; у броней «без возврата» — 0 (депозит уже получен)."""
    return df["booking_value"].where(df["deposit_type"] != NON_REFUND, 0.0).to_numpy(dtype=float)


def losses(y: pd.Series, proba: np.ndarray, value: np.ndarray, threshold: float,
           assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> dict[str, float]:
    """Потери при правиле «предоплата, если вероятность отмены ≥ порога»: пропущенные отмены и лишние предоплаты."""
    canceled = np.asarray(y) == 1
    prepay = proba >= threshold
    missed = float(value[canceled & ~prepay].sum() * assumptions.unsold_share)
    false_alarm = float(value[~canceled & prepay].sum() * assumptions.loss_prob)
    return {"missed": missed, "false_alarm": false_alarm, "total": missed + false_alarm}


def loss_curve(y: pd.Series, proba: np.ndarray, value: np.ndarray,
               assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> pd.DataFrame:
    """Потери для каждого порога из THRESHOLDS."""
    rows = [{"threshold": t, **losses(y, proba, value, t, assumptions)} for t in THRESHOLDS]
    return pd.DataFrame(rows).set_index("threshold")


def best_threshold(curve: pd.DataFrame) -> float:
    """Порог с минимальными суммарными потерями (при равенстве — наименьший)."""
    return float(curve["total"].idxmin())


def threshold_sensitivity(y: pd.Series, proba: np.ndarray, value: np.ndarray, unsold_grid: list[float],
                          loss_grid: list[float]) -> pd.DataFrame:
    """Оптимальный порог для каждой пары допущений: строки — доля неперепроданного, колонки — вероятность ухода."""
    return pd.DataFrame({
        loss_prob: {
            unsold: best_threshold(loss_curve(y, proba, value, Assumptions(unsold, loss_prob)))
            for unsold in unsold_grid
        } for loss_prob in loss_grid
    })


def threshold_report(y: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Точность, полнота и доля броней, с которых берём предоплату, при выбранном пороге."""
    prepay = proba >= threshold
    return {
        "precision": float(precision_score(y, prepay, zero_division=0)),
        "recall": float(recall_score(y, prepay, zero_division=0)),
        "prepay_share": float(prepay.mean()),
    }
