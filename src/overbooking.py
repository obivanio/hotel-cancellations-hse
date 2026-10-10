"""Овербукинг: сколько отмен ждать на каждую дату заезда и сколько номеров можно продать сверх броней."""
import numpy as np
import pandas as pd

from src.pipeline import ROOT
from src.policies import ALL_HOTELS

OVERBOOKING_PATH = ROOT / "reports" / "overbooking.csv"
OVERBOOK_CONFIDENCE = 0.9
WALK_COSTS = (0.5, 1.0, 2.0)


def cancel_distribution(proba: np.ndarray) -> np.ndarray:
    """Распределение числа отмен среди броней с разными вероятностями отмены (пуассон-биномиальное).

    Отмены считаются независимыми; элемент k — вероятность ровно k отмен. Считается свёрткой по броням.
    """
    dist = np.array([1.0])
    for p in np.asarray(proba, dtype=float):
        dist = np.convolve(dist, [1.0 - p, p])
    return dist


def cautious_count(proba: np.ndarray, confidence: float = OVERBOOK_CONFIDENCE) -> int:
    """Наибольшее число отмен k, которое наступит с вероятностью не ниже confidence: P(отмен ≥ k) ≥ confidence."""
    at_least = np.cumsum(cancel_distribution(proba)[::-1])[::-1]
    return int(np.flatnonzero(at_least >= confidence - 1e-12).max())


def benchmark_rates(train: pd.DataFrame) -> pd.Series:
    """Бенчмарк без модели: доля отмен по ячейке «месяц заезда × день недели» на обучении."""
    return train.groupby(["arrival_month", "arrival_weekday"])["is_canceled"].mean()


def benchmark_proba(rates: pd.Series, df: pd.DataFrame, fallback: float) -> np.ndarray:
    """Вероятность отмены каждой брони по бенчмарку; новая ячейка — общая доля отмен обучения (fallback)."""
    keys = pd.MultiIndex.from_arrays([df["arrival_month"], df["arrival_weekday"]])
    return rates.reindex(keys).fillna(fallback).to_numpy(dtype=float)


def date_table(df: pd.DataFrame, proba: np.ndarray, bench: np.ndarray,
               confidence: float = OVERBOOK_CONFIDENCE) -> pd.DataFrame:
    """По каждой дате заезда (все отели и каждый отель): брони, факт отмен, ожидание и осторожная оценка.

    Ожидание — сумма вероятностей; осторожная оценка — cautious_count; то же по бенчмарку (`bench_*`);
    mean_value — средняя стоимость брони даты.
    """
    frame = pd.DataFrame({
        "hotel": df["hotel"].to_numpy(dtype=str), "arrival_date": df["arrival_date"].to_numpy(),
        "canceled": df["is_canceled"].to_numpy(), "value": df["booking_value"].to_numpy(dtype=float),
        "proba": np.asarray(proba, dtype=float), "bench": np.asarray(bench, dtype=float),
    })
    rows = []
    for hotel, part in [(ALL_HOTELS, frame)] + list(frame.groupby("hotel")):
        for date, day in part.groupby("arrival_date"):
            rows.append({
                "hotel": hotel, "arrival_date": date, "bookings": len(day), "actual": int(day["canceled"].sum()),
                "expected": float(day["proba"].sum()), "cautious": cautious_count(day["proba"], confidence),
                "bench_expected": float(day["bench"].sum()), "bench_cautious": cautious_count(day["bench"], confidence),
                "mean_value": float(day["value"].mean()),
            })
    return pd.DataFrame(rows)


def overbooking_outcome(table: pd.DataFrame, plan: str, walk_cost: float = 1.0) -> dict[str, float]:
    """Итог овербукинга по датам: план — колонка с числом номеров, проданных сверх броней.

    Перепродано min(план, факт отмен); если план > факт, лишним гостям не хватает номера — каждому платим
    стоимость брони × walk_cost (допущение). Деньги: перепроданные номера × средняя стоимость минус «гости без номера».
    """
    planned, actual, value = table[plan].to_numpy(), table["actual"].to_numpy(), table["mean_value"].to_numpy()
    resold = np.minimum(planned, actual)
    walked = np.clip(planned - actual, 0, None)
    gain, cost = float((resold * value).sum()), float((walked * value).sum() * walk_cost)
    return {
        "resold": float(resold.sum()), "short_dates": float((walked > 0).sum()), "walked": float(walked.sum()),
        "gain": gain, "walk_cost": cost, "net": gain - cost,
    }


def outcome_table(table: pd.DataFrame, walk_costs: tuple[float, ...] = WALK_COSTS) -> pd.DataFrame:
    """Итоги овербукинга для модели и бенчмарка (осторожные оценки) при разной цене «гостя без номера».

    Решение принимает каждый отель отдельно, поэтому считаются строки отелей (без сводных «все отели»).
    """
    hotels = table[table["hotel"] != ALL_HOTELS]
    rows = [{"plan": name, "walk_cost_share": cost, **overbooking_outcome(hotels, plan, cost)}
            for name, plan in {"модель": "cautious", "бенчмарк": "bench_cautious"}.items() for cost in walk_costs]
    return pd.DataFrame(rows)


def weekly_table(dates: pd.DataFrame, hotel: str = ALL_HOTELS) -> pd.DataFrame:
    """Отмены по календарным неделям (пн–вс) для одного отеля (по умолчанию «все отели»).

    Строки таблицы `date_table` с датой заезда в одной неделе суммируются. Колонки: week_start и week_end —
    первая и последняя даты недели, которые есть в таблице (у неполной недели это её реальный срок);
    actual, expected, bench_expected — суммы факта, ожидания модели и ожидания бенчмарка за неделю.
    """
    part = dates[dates["hotel"] == hotel].copy()
    dt = pd.to_datetime(part["arrival_date"])
    part["week"] = dt.dt.to_period("W-SUN")
    part["arrival_date"] = dt
    return (part.groupby("week").agg(week_start=("arrival_date", "min"), week_end=("arrival_date", "max"),
                                     actual=("actual", "sum"), expected=("expected", "sum"),
                                     bench_expected=("bench_expected", "sum")).reset_index(drop=True))
