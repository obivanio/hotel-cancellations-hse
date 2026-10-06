"""Правила предоплаты: маски «с какой брони брать предоплату», деньги и задетые гости, подбор и сравнение правил."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation import DEFAULT_ASSUMPTIONS, Assumptions, value_at_risk
from src.pipeline import ROOT

REPORTS_DIR = ROOT / "reports"
POLICIES_PATH = REPORTS_DIR / "policies.csv"
CURVES_PATH = REPORTS_DIR / "share_curves.csv"
ALL_HOTELS = "все отели"
LEAD_DAYS = range(0, 366)
TOP_SHARES = np.round(np.linspace(0.1, 1.0, 10), 2)
SHARES = np.round(np.linspace(0.0, 1.0, 21), 2)
BASELINE = "как сейчас"


@dataclass(frozen=True)
class Scored:
    """Брони одной части выборки со всем, что нужно правилам: факт отмены, риск модели, срок, деньги, отель."""

    canceled: np.ndarray
    proba: np.ndarray
    lead_time: np.ndarray
    value: np.ndarray
    hotel: np.ndarray

    def subset(self, mask: np.ndarray) -> "Scored":
        """Брони, отобранные маской (например, один отель)."""
        return Scored(self.canceled[mask], self.proba[mask], self.lead_time[mask], self.value[mask],
                      self.hotel[mask])


def scored(df: pd.DataFrame, proba: np.ndarray) -> Scored:
    """Собрать Scored из таблицы броней и калиброванных вероятностей отмены."""
    return Scored(
        canceled=df["is_canceled"].to_numpy() == 1,
        proba=np.asarray(proba, dtype=float),
        lead_time=df["lead_time"].to_numpy(dtype=float),
        value=value_at_risk(df),
        hotel=df["hotel"].to_numpy(dtype=str),
    )


def top_share_mask(scores: np.ndarray, share: float) -> np.ndarray:
    """Ровно round(share × n) броней с наибольшим скором; при равных скорах — те, что раньше в таблице."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:int(round(share * len(order)))]] = True
    return mask


def policy_metrics(s: Scored, prepay: np.ndarray, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> dict[str, float]:
    """Потери правила (как в шаге 3) и его цена для гостей: доля предоплат, задетых надёжных, пойманных отмен."""
    missed = float(s.value[s.canceled & ~prepay].sum() * assumptions.unsold_share)
    false_alarm = float(s.value[~s.canceled & prepay].sum() * assumptions.loss_prob)
    reliable, canceled = (~s.canceled).sum(), s.canceled.sum()
    return {
        "missed": missed,
        "false_alarm": false_alarm,
        "total": missed + false_alarm,
        "prepay_share": float(prepay.mean()),
        "touched_reliable": float((prepay & ~s.canceled).sum() / reliable) if reliable else 0.0,
        "caught_cancels": float((prepay & s.canceled).sum() / canceled) if canceled else 0.0,
    }


def best_lead_days(s: Scored, assumptions: Assumptions = DEFAULT_ASSUMPTIONS, grid: range = LEAD_DAYS) -> int:
    """Правило «срок брони > N дней»: N с минимальными потерями (при равенстве — наименьший)."""
    totals = [policy_metrics(s, s.lead_time > days, assumptions)["total"] for days in grid]
    return grid[int(np.argmin(totals))]


def policy_masks(s: Scored, lead_days: int, threshold: float,
                 top_shares: np.ndarray = TOP_SHARES) -> dict[str, np.ndarray]:
    """Маски предоплаты пяти правил: как сейчас, со всех, по сроку, модель по порогу, модель топ-X %."""
    masks = {
        BASELINE: np.zeros(len(s.proba), dtype=bool),
        "со всех": np.ones(len(s.proba), dtype=bool),
        f"срок брони > {lead_days} дн.": s.lead_time > lead_days,
        f"модель, порог {threshold:.2f}".replace(".", ","): s.proba >= threshold,
    }
    for share in top_shares:
        masks[f"модель, топ-{share * 100:.0f} %"] = top_share_mask(s.proba, share)
    return masks


def hotel_parts(s: Scored) -> dict[str, Scored]:
    """Все брони и брони каждого отеля отдельно."""
    return {ALL_HOTELS: s} | {hotel: s.subset(s.hotel == hotel) for hotel in sorted(set(s.hotel))}


def policy_table(s: Scored, lead_days: int, threshold: float,
                 assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> pd.DataFrame:
    """Показатели всех правил для всех броней и по отелям; экономия — против «как сейчас» в том же отеле.

    Правила применяются внутри каждого отеля: «топ-X %» — это X % броней именно этого отеля.
    """
    rows = []
    for hotel, part in hotel_parts(s).items():
        for policy, prepay in policy_masks(part, lead_days, threshold).items():
            rows.append({"hotel": hotel, "policy": policy, **policy_metrics(part, prepay, assumptions)})
    table = pd.DataFrame(rows)
    baseline = table[table["policy"] == BASELINE].set_index("hotel")["total"]
    table["saving"] = table["hotel"].map(baseline) - table["total"]
    return table


def share_curve(s: Scored, scores: np.ndarray, shares: np.ndarray = SHARES,
                assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> pd.DataFrame:
    """Потери и цена для гостей, если брать предоплату с доли броней с наибольшим скором (индекс — доля)."""
    rows = [{"share": share, **policy_metrics(s, top_share_mask(scores, share), assumptions)} for share in shares]
    return pd.DataFrame(rows).set_index("share")


def equal_share_curves(s: Scored, shares: np.ndarray = SHARES,
                       assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> pd.DataFrame:
    """Сравнение при равной доле предоплаты: брони выбирает модель (по риску) или срок брони (самые ранние).

    Считается для всех броней и по отелям; колонки: hotel, method, share и показатели policy_metrics.
    """
    frames = []
    for hotel, part in hotel_parts(s).items():
        for method, scores in {"модель": part.proba, "срок брони": part.lead_time}.items():
            curve = share_curve(part, scores, shares, assumptions).reset_index()
            frames.append(curve.assign(hotel=hotel, method=method))
    columns = ["hotel", "method", "share"]
    table = pd.concat(frames, ignore_index=True)
    return table[columns + [column for column in table.columns if column not in columns]]


def policy_sensitivity(valid: Scored, test: Scored, unsold_grid: list[float],
                       loss_grid: list[float]) -> pd.DataFrame:
    """Лучшее правило на тесте при каждой паре допущений.

    Для каждой пары заново подбираются N по валидации и порог модели loss_prob / (unsold_share + loss_prob)
    (оптимальный для калиброванных вероятностей); сравниваются «как сейчас», «со всех», срок и модель.
    """
    rows = []
    for unsold in unsold_grid:
        for loss_prob in loss_grid:
            assumptions = Assumptions(unsold, loss_prob)
            lead_days = best_lead_days(valid, assumptions)
            threshold = loss_prob / (unsold + loss_prob)
            totals = {
                BASELINE: policy_metrics(test, np.zeros(len(test.proba), dtype=bool), assumptions)["total"],
                "со всех": policy_metrics(test, np.ones(len(test.proba), dtype=bool), assumptions)["total"],
                "срок брони": policy_metrics(test, test.lead_time > lead_days, assumptions)["total"],
                "модель": policy_metrics(test, test.proba >= threshold, assumptions)["total"],
            }
            best = min(totals, key=totals.get)
            rows.append({"unsold_share": unsold, "loss_prob": loss_prob, "lead_days": lead_days,
                         "threshold": threshold, **totals, "best": best, "best_total": totals[best]})
    return pd.DataFrame(rows)
