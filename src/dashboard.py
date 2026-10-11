"""Эталонные расчёты дашборда: срез броней, очередь по риску, потери и счётчики, кривые, тепловая карта, овербукинг.

Браузер (src/dashboard_assets/dashboard.js) повторяет эти формулы, тест сверки проверяет совпадение. Суммы денег
считаются накопительно (np.cumsum) в том же порядке, что и в браузере, поэтому цифры совпадают до последнего знака.
"""
import numpy as np
import pandas as pd

from src.evaluation import DEFAULT_ASSUMPTIONS, Assumptions, value_at_risk
from src.overbooking import overbooking_outcome, week_labels
from src.policies import ALL_HOTELS, REPORTS_DIR, Scored
from src.report import MONTH_NAMES

SCORED_TEST_PATH = REPORTS_DIR / "scored_test.csv"
SCORED_COLUMNS = ["hotel", "market_segment", "arrival_month", "lead_time", "value", "is_canceled", "proba"]
HOTEL_NAMES = {ALL_HOTELS: "Оба отеля", "City Hotel": "Городской", "Resort Hotel": "Курортный"}
SEGMENT_NAMES = {
    "Online TA": "Онлайн-агентства", "Offline TA/TO": "Офлайн-агентства", "Groups": "Группы", "Direct": "Прямые",
    "Corporate": "Корпоративные", "Complementary": "Бесплатные", "Aviation": "Авиакомпании",
}
BUCKETS = 100
SMALL_MONEY = 100_000


def scored_frame(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Таблица для дашборда: бронь → отель, сегмент, месяц, срок, стоимость при отмене, факт отмены, риск модели.

    value — стоимость, которую отель теряет при отмене (у броней «без возврата» — 0), как в правилах шага 4.
    """
    return pd.DataFrame({
        "hotel": df["hotel"].astype(str).to_numpy(), "market_segment": df["market_segment"].astype(str).to_numpy(),
        "arrival_month": df["arrival_month"].to_numpy(dtype=int), "lead_time": df["lead_time"].to_numpy(dtype=float),
        "value": value_at_risk(df), "is_canceled": df["is_canceled"].to_numpy(dtype=int),
        "proba": np.asarray(proba, dtype=float),
    })


def filter_frame(frame: pd.DataFrame, hotel: str, segments: list[str], months: list[int]) -> pd.DataFrame:
    """Брони выбранного отеля («все отели» — оба), выбранных сегментов и месяцев заезда."""
    mask = frame["market_segment"].isin(segments) & frame["arrival_month"].isin(months)
    if hotel != ALL_HOTELS:
        mask &= frame["hotel"] == hotel
    return frame[mask]


def to_scored(frame: pd.DataFrame) -> Scored:
    """Scored из таблицы scored_frame — вход функций src/policies.py."""
    return Scored(canceled=frame["is_canceled"].to_numpy() == 1, proba=frame["proba"].to_numpy(dtype=float),
                  lead_time=frame["lead_time"].to_numpy(dtype=float), value=frame["value"].to_numpy(dtype=float),
                  hotel=frame["hotel"].to_numpy(dtype=str))


def model_threshold(assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> float:
    """Порог модели loss_prob / (unsold_share + loss_prob): брать предоплату, когда ожидаемая потеря от отмены
    больше ожидаемой потери от ушедшего гостя (для калиброванных вероятностей)."""
    return assumptions.loss_prob / (assumptions.unsold_share + assumptions.loss_prob)


def cut_size(bookings: int, pct: int) -> int:
    """Сколько броней в первых pct % очереди: pct × bookings / 100, округление половины вверх, в целых числах."""
    return (pct * bookings + 50) // 100


def queue_order(frame: pd.DataFrame, column: str = "proba") -> np.ndarray:
    """Порядок броней в очереди: по убыванию column, при равенстве — как в таблице (как top_share_mask)."""
    return np.argsort(-frame[column].to_numpy(dtype=float), kind="stable")


def _prefix(frame: pd.DataFrame, order: np.ndarray) -> dict[str, np.ndarray]:
    """Накопленные по очереди суммы: стоимость отменённых и надёжных броней, число тех и других (value > 0).

    Элемент k — сумма по первым k броням очереди (элемент 0 — ноль).
    """
    value = frame["value"].to_numpy(dtype=float)[order]
    canceled = frame["is_canceled"].to_numpy()[order] == 1
    counted = value > 0

    def acc(values: np.ndarray) -> np.ndarray:
        """Накопленная сумма с нулём в начале."""
        return np.concatenate([[0.0], np.cumsum(values)])

    return {
        "cancel_value": acc(np.where(canceled, value, 0.0)), "reliable_value": acc(np.where(canceled, 0.0, value)),
        "cancels": acc((canceled & counted).astype(float)), "reliable": acc((~canceled & counted).astype(float)),
    }


def _outcome(prefix: dict[str, np.ndarray], k: int, assumptions: Assumptions) -> dict[str, float]:
    """Потери и счётчики, если предоплата взята с первых k броней очереди.

    Пойманные, пропущенные отмены и задетые надёжные гости считаются среди броней с value > 0, как в
    policy_metrics: брони «без возврата» уже оплачены, требование предоплаты для них ничего не меняет.
    """
    cancels, reliable = prefix["cancels"], prefix["reliable"]
    missed = assumptions.unsold_share * (prefix["cancel_value"][-1] - prefix["cancel_value"][k])
    false_alarm = assumptions.loss_prob * prefix["reliable_value"][k]
    missed_cancels = cancels[-1] - cancels[k]
    return {
        "missed": missed, "false_alarm": false_alarm, "total": missed + false_alarm,
        "caught": cancels[k], "touched": reliable[k], "missed_cancels": missed_cancels,
        "caught_share": cancels[k] / cancels[-1] if cancels[-1] else 0.0,
        "touched_share": reliable[k] / reliable[-1] if reliable[-1] else 0.0,
        "missed_share": missed_cancels / cancels[-1] if cancels[-1] else 0.0,
    }


def queue_outcome(frame: pd.DataFrame, pct: int, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> dict[str, float]:
    """Итог черты «предоплата с pct % броней»: потери, пойманные и пропущенные отмены, задетые надёжные гости.

    Пустой срез — все цифры 0 (без NaN).
    """
    if frame.empty:
        return {key: 0.0 for key in ("missed", "false_alarm", "total", "caught", "touched", "missed_cancels",
                                     "caught_share", "touched_share", "missed_share")}
    return _outcome(_prefix(frame, queue_order(frame)), cut_size(len(frame), pct), assumptions)


def threshold_pct(frame: pd.DataFrame, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> int:
    """Доля броней (целые %) с риском не ниже порога model_threshold — где стоит отметка порога на очереди."""
    if frame.empty:
        return 0
    above = int((frame["proba"].to_numpy(dtype=float) >= model_threshold(assumptions)).sum())
    return (200 * above + len(frame)) // (2 * len(frame))


def key_numbers(frame: pd.DataFrame, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> dict[str, float]:
    """Опорные цифры среза: число броней, порог и его место в очереди, потери без предоплаты и со всех."""
    return {
        "bookings": len(frame), "threshold": model_threshold(assumptions),
        "threshold_pct": threshold_pct(frame, assumptions),
        "baseline": queue_outcome(frame, 0, assumptions)["total"],
        "everyone": queue_outcome(frame, 100, assumptions)["total"],
    }


def queue_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    """100 полосок очереди: каждая — 1 % броней среза по убыванию риска.

    Границы полоски i — cut_size(n, i) и cut_size(n, i + 1). Колонки: bookings, cancel_share (доля отмен,
    сглаженная внутри групп одинакового риска: модель не различает такие брони, поэтому каждая показывает общую
    долю отмен своей группы), mean_risk (средний риск модели); у пустой полоски все три — 0.
    """
    order = queue_order(frame)
    smoothed = frame.groupby("proba")["is_canceled"].transform("mean").to_numpy(dtype=float)
    canceled = np.concatenate([[0.0], np.cumsum(smoothed[order])])
    risk = np.concatenate([[0.0], np.cumsum(frame["proba"].to_numpy(dtype=float)[order])])
    rows = []
    for i in range(BUCKETS):
        start, end = cut_size(len(frame), i), cut_size(len(frame), i + 1)
        size = end - start
        rows.append({"bookings": size,
                     "cancel_share": (canceled[end] - canceled[start]) / size if size else 0.0,
                     "mean_risk": (risk[end] - risk[start]) / size if size else 0.0})
    return pd.DataFrame(rows)


def share_curves(frame: pd.DataFrame, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> pd.DataFrame:
    """Доля броней с предоплатой (0–100 %, шаг 1 %) → потери, когда брони выбирает модель или срок брони.

    Колонки: method («модель» — по риску, «срок брони» — самые ранние), pct, total, touched_share,
    caught_share. Пустой срез — пустая таблица.
    """
    columns = ["method", "pct", "total", "touched_share", "caught_share"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for method, column in (("модель", "proba"), ("срок брони", "lead_time")):
        prefix = _prefix(frame, queue_order(frame, column))
        for pct in range(BUCKETS + 1):
            rows.append({"method": method, "pct": pct,
                         **_outcome(prefix, cut_size(len(frame), pct), assumptions)})
    return pd.DataFrame(rows)[columns]


def risk_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Сегмент × месяц заезда: средний риск модели (%), доля отмен на деле (%) и число броней в клетке."""
    grouped = frame.groupby(["market_segment", "arrival_month"])
    tables = {
        "risk": grouped["proba"].mean() * 100,
        "cancel": grouped["is_canceled"].mean() * 100,
        "bookings": grouped.size(),
    }
    return {name: table.unstack().rename(columns=MONTH_NAMES) for name, table in tables.items()}


def _hotel_rows(dates: pd.DataFrame, hotel: str) -> pd.DataFrame:
    """Строки таблицы овербукинга выбранного отеля; «все отели» — строки двух отелей, сводные строки не берутся."""
    if hotel == ALL_HOTELS:
        return dates[dates["hotel"] != ALL_HOTELS]
    return dates[dates["hotel"] == hotel]


def weekly_overbooking(dates: pd.DataFrame, hotel: str) -> pd.DataFrame:
    """По календарным неделям заезда (пн–вс): отмены на деле и сколько номеров продаём сверх брони.

    «Все отели» — сумма решений двух отелей (как в ноутбуке 04), сводные строки таблицы не используются.
    Колонки: label («3–7 мая»), month (месяц начала недели), actual, cautious (осторожная оценка модели),
    bench_cautious (бенчмарк без модели).
    """
    part = _hotel_rows(dates, hotel).assign(arrival_date=lambda d: pd.to_datetime(d["arrival_date"]))
    weeks = (part.groupby(part["arrival_date"].dt.to_period("W-SUN"))
             .agg(week_start=("arrival_date", "min"), week_end=("arrival_date", "max"), actual=("actual", "sum"),
                  cautious=("cautious", "sum"), bench_cautious=("bench_cautious", "sum"))
             .reset_index(drop=True))
    weeks = weeks.assign(label=week_labels(weeks), month=weeks["week_start"].dt.month)
    return weeks[["label", "month", "actual", "cautious", "bench_cautious"]]


def overbooking_totals(dates: pd.DataFrame, hotel: str) -> dict[str, float]:
    """Итог осторожной оценки по датам отеля: перепродано номеров, гостей без номера, доля дат без нехватки.

    «Все отели» — сумма решений двух отелей (как в ноутбуке 04), сводные строки таблицы не используются; dates —
    число строк «отель × дата». Дата без нехватки — осторожная оценка не больше отмен на деле; covered_bench —
    то же у бенчмарка. per_day — сколько номеров продаём сверх брони на одну дату заезда в среднем (у «все отели» —
    сумма двух отелей на дату), per_day_min и per_day_max — меньше и больше всего по датам.
    """
    rows = _hotel_rows(dates, hotel)
    outcome = overbooking_outcome(rows, "cautious")
    per_day = rows.groupby("arrival_date")["cautious"].sum()
    return {
        "dates": len(rows), "resold": outcome["resold"], "walked": outcome["walked"],
        "per_day": float(per_day.mean()) if len(per_day) else 0.0,
        "per_day_min": float(per_day.min()) if len(per_day) else 0.0,
        "per_day_max": float(per_day.max()) if len(per_day) else 0.0,
        "covered": float((rows["cautious"] <= rows["actual"]).mean()) if len(rows) else 0.0,
        "covered_bench": float((rows["bench_cautious"] <= rows["actual"]).mean()) if len(rows) else 0.0,
    }


def verdict_numbers(policies: pd.DataFrame, dates: pd.DataFrame) -> dict[str, float]:
    """Цифры блока «Что делать бизнесу» при допущениях по умолчанию: правила шага 4 и овербукинг по отелям."""
    rows = policies[policies["hotel"] == ALL_HOTELS].set_index("policy")
    threshold = rows.loc[[name for name in rows.index if name.startswith("модель, порог")][0]]
    top30 = rows.loc["модель, топ-30 %"]
    return {
        "baseline": rows.loc["как сейчас", "total"], "everyone": rows.loc["со всех", "total"],
        "model": threshold["total"], "model_prepay": threshold["prepay_share"],
        "model_touched": threshold["touched_reliable"], "model_caught": threshold["caught_cancels"],
        "top30": top30["total"], "top30_touched": top30["touched_reliable"], "top30_caught": top30["caught_cancels"],
        **{f"covered_{key}": overbooking_totals(dates, hotel)["covered"]
           for key, hotel in (("all", ALL_HOTELS), ("city", "City Hotel"), ("resort", "Resort Hotel"))},
    }


def comma(value: float, digits: int = 2) -> str:
    """Число с десятичной запятой: 0,17 (без «-0,00» у малых отрицательных)."""
    return f"{round(value, digits) + 0.0:.{digits}f}".replace(".", ",")


def money(value: float) -> str:
    """Сумма с десятичной запятой: от 0,1 млн € — в миллионах (2,11 млн €), меньше — в тысячах (23,4 тыс. €)."""
    if abs(value) < SMALL_MONEY:
        return f"{comma(value / 1e3, 1)} тыс. €"
    return f"{comma(value / 1e6)} млн €"


def count(value: float) -> str:
    """Целое число с пробелом между тысячами: 4 898."""
    return f"{value:,.0f}".replace(",", " ")


def percent(share: float) -> str:
    """Доля в процентах без знаков после запятой: 60 %."""
    return f"{share * 100:.0f} %"
