"""Расчёты и графики дашборда: срез броней по фильтрам, ключевые цифры, правила, овербукинг и риск по сегментам."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.evaluation import DEFAULT_ASSUMPTIONS, Assumptions, value_at_risk
from src.overbooking import overbooking_outcome, weekly_table
from src.policies import ALL_HOTELS, BASELINE, REPORTS_DIR, SHARES, Scored, policy_masks, policy_metrics, share_curve
from src.report import LAYOUT, MONTH_NAMES, week_traces

SCORED_TEST_PATH = REPORTS_DIR / "scored_test.csv"
SCORED_VALID_PATH = REPORTS_DIR / "scored_valid.csv"
SCORED_COLUMNS = ["hotel", "market_segment", "arrival_month", "lead_time", "value", "is_canceled", "proba"]
HOTELS = {"оба отеля": ALL_HOTELS, "городской": "City Hotel", "курортный": "Resort Hotel"}
RULE_COLUMNS = {
    "policy": "правило", "total": "потери, млн €", "saving": "экономия против «как сейчас», млн €",
    "prepay_share": "предоплата, % броней", "touched_reliable": "задето надёжных гостей, %",
    "caught_cancels": "поймано отмен, %",
}


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


def key_numbers(frame: pd.DataFrame, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> dict[str, float]:
    """Ключевые цифры среза: потери без предоплаты, со всех и по модели, экономия модели, задетые надёжные гости.

    У пустого среза все цифры — 0 (без NaN); bookings показывает, сколько броней в срезе.
    """
    threshold = model_threshold(assumptions)
    if frame.empty:
        return {"bookings": 0, "threshold": threshold, "baseline": 0.0, "everyone": 0.0, "model": 0.0,
                "saving": 0.0, "touched_reliable": 0.0, "prepay_share": 0.0}
    s = to_scored(frame)
    model = policy_metrics(s, s.proba >= threshold, assumptions)
    everyone = policy_metrics(s, np.ones(len(frame), dtype=bool), assumptions)
    return {
        "bookings": len(frame), "threshold": threshold,
        "baseline": policy_metrics(s, np.zeros(len(frame), dtype=bool), assumptions)["total"],
        "everyone": everyone["total"], "model": model["total"], "saving": everyone["total"] - model["total"],
        "touched_reliable": model["touched_reliable"], "prepay_share": model["prepay_share"],
    }


def rules_table(frame: pd.DataFrame, lead_days: int, share: float,
                assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> pd.DataFrame:
    """Пять правил на срезе: как сейчас, со всех, срок брони > N, порог модели по формуле, модель топ-share.

    Колонки: policy, missed, false_alarm, total, prepay_share, touched_reliable, caught_cancels, saving (против
    «как сейчас»). Пустой срез — пустая таблица с теми же колонками.
    """
    columns = ["policy", "missed", "false_alarm", "total", "prepay_share", "touched_reliable", "caught_cancels",
               "saving"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    s = to_scored(frame)
    masks = policy_masks(s, lead_days, model_threshold(assumptions), top_shares=np.array([share]))
    table = pd.DataFrame([{"policy": name, **policy_metrics(s, prepay, assumptions)} for name, prepay in masks.items()])
    table["saving"] = table.loc[table["policy"] == BASELINE, "total"].iloc[0] - table["total"]
    return table[columns]


def share_curves(frame: pd.DataFrame, assumptions: Assumptions = DEFAULT_ASSUMPTIONS,
                 shares: np.ndarray = SHARES) -> pd.DataFrame:
    """Доля броней с предоплатой → потери, когда брони выбирает модель (по риску) или срок брони (самые ранние).

    Колонки: method, share, total, touched_reliable, caught_cancels. Пустой срез — пустая таблица.
    """
    columns = ["method", "share", "total", "touched_reliable", "caught_cancels"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    s = to_scored(frame)
    curves = [share_curve(s, scores, shares, assumptions).reset_index().assign(method=method)
              for method, scores in {"модель": s.proba, "срок брони": s.lead_time}.items()]
    return pd.concat(curves, ignore_index=True)[columns]


def _in_months(dates: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """Строки таблицы овербукинга с датой заезда в выбранных месяцах."""
    return dates[pd.to_datetime(dates["arrival_date"]).dt.month.isin(months)]


def dates_slice(dates: pd.DataFrame, hotel: str, months: list[int]) -> pd.DataFrame:
    """Строки таблицы овербукинга для отеля («все отели» — сводные строки) и месяцев заезда."""
    return _in_months(dates[dates["hotel"] == hotel], months)


def overbooking_summary(dates: pd.DataFrame, hotel: str, months: list[int],
                        walk_cost: float = 1.0) -> dict[str, dict[str, float]]:
    """Итог осторожной оценки модели и бенчмарка для отеля и месяцев: перепродано, даты и гости без номера, деньги.

    Решение принимает каждый отель, поэтому для «все отели» складываются строки двух отелей, а не сводные.
    """
    rows = dates[dates["hotel"] != ALL_HOTELS] if hotel == ALL_HOTELS else dates[dates["hotel"] == hotel]
    rows = _in_months(rows, months)
    return {name: overbooking_outcome(rows, plan, walk_cost)
            for name, plan in {"модель": "cautious", "бенчмарк": "bench_cautious"}.items()}


def risk_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Сегмент × месяц заезда: средний риск модели (%), доля отмен на деле (%) и число броней в клетке."""
    grouped = frame.groupby(["market_segment", "arrival_month"])
    tables = {
        "risk": grouped["proba"].mean() * 100,
        "cancel": grouped["is_canceled"].mean() * 100,
        "bookings": grouped.size(),
    }
    return {name: table.unstack().rename(columns=MONTH_NAMES) for name, table in tables.items()}


def money(value: float) -> str:
    """Сумма в миллионах евро с десятичной запятой: 2,11 млн €."""
    return f"{value / 1e6:.2f} млн €".replace(".", ",")


def count(value: float) -> str:
    """Целое число с пробелом между тысячами: 4 898."""
    return f"{value:,.0f}".replace(",", " ")


def percent(share: float) -> str:
    """Доля в процентах без знаков после запятой: 60 %."""
    return f"{share * 100:.0f} %"


def display_rules(table: pd.DataFrame) -> pd.DataFrame:
    """Таблица правил для показа: деньги в млн € и доли в % строками с десятичной запятой, русские заголовки."""
    shown = table[list(RULE_COLUMNS)].copy()
    for column in ("total", "saving"):
        shown[column] = shown[column].map(lambda value: f"{value / 1e6:.3f}".replace(".", ","))
    for column in ("prepay_share", "touched_reliable", "caught_cancels"):
        shown[column] = shown[column].map(percent)
    return shown.rename(columns=RULE_COLUMNS)


def fig_share_curves(curves: pd.DataFrame, share: float) -> go.Figure:
    """Кривые «доля броней с предоплатой → потери» для модели и срока брони и отметка выбранной доли."""
    fig = go.Figure()
    for method, dash in (("модель", "solid"), ("срок брони", "dash")):
        line = curves[curves["method"] == method]
        fig.add_trace(go.Scatter(
            x=line["share"] * 100, y=line["total"] / 1e6, mode="lines+markers", name=f"брони выбирает {method}",
            line={"dash": dash}, customdata=line["touched_reliable"] * 100,
            hovertemplate="предоплата с %{x:.0f} % броней<br>потери %{y:.2f} млн €"
                          "<br>задето надёжных %{customdata:.0f} %<extra></extra>"))
    fig.add_vline(x=share * 100, line={"color": "grey", "dash": "dot"},
                  annotation_text=f"выбрано: {percent(share)}")
    fig.update_layout(**LAYOUT, xaxis_title="доля броней с предоплатой, %", yaxis_title="потери, млн €",
                      legend={"orientation": "h", "y": -0.25})
    return fig


def fig_weeks(dates: pd.DataFrame, hotel: str) -> go.Figure:
    """По неделям заезда для одного отеля: столбики фактических отмен, линии ожидания модели и бенчмарка."""
    fig = go.Figure(week_traces(weekly_table(dates, hotel)))
    fig.update_layout(**LAYOUT, xaxis_title="неделя заезда", yaxis_title="отмен за неделю", xaxis_tickangle=-45,
                      legend={"orientation": "h", "y": -0.45})
    return fig


def fig_heatmap(pivot: pd.DataFrame, counts: pd.DataFrame, title: str) -> go.Figure:
    """Тепловая карта «сегмент × месяц» в процентах; в подсказке — число броней в клетке."""
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=list(pivot.columns), y=list(pivot.index), colorscale="Blues", zmin=0, zmax=100,
        text=pivot.round(0).to_numpy(), texttemplate="%{text:.0f}", customdata=counts.to_numpy(),
        hovertemplate="%{y}, %{x}: %{z:.0f} %<br>броней: %{customdata:.0f}<extra></extra>",
        colorbar={"title": "%"}))
    fig.update_layout(**LAYOUT, title=title, xaxis_title="месяц заезда", yaxis_title="сегмент")
    return fig
