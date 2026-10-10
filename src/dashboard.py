"""Расчёты и графики дашборда: срез броней по фильтрам, ключевые цифры, правила, овербукинг и риск по сегментам."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.evaluation import DEFAULT_ASSUMPTIONS, Assumptions, value_at_risk
from src.overbooking import overbooking_outcome, weekly_table
from src.policies import ALL_HOTELS, BASELINE, REPORTS_DIR, SHARES, Scored, policy_masks, policy_metrics, share_curve
from src.report import MONTH_NAMES, week_traces
from src.theme import COBALT, HOTEL_COLORS, INK, LAYOUT, OCHRE, STONE, one_color_scale

SCORED_TEST_PATH = REPORTS_DIR / "scored_test.csv"
SCORED_VALID_PATH = REPORTS_DIR / "scored_valid.csv"
SCORED_COLUMNS = ["hotel", "market_segment", "arrival_month", "lead_time", "value", "is_canceled", "proba"]
HOTELS = {"оба отеля": ALL_HOTELS, "городской": "City Hotel", "курортный": "Resort Hotel"}
RULE_COLUMNS = {
    "policy": "правило", "total": "потери, {unit}", "saving": "экономия против «как сейчас», {unit}",
    "prepay_share": "предоплата, % броней", "touched_reliable": "задето надёжных гостей, %",
    "caught_cancels": "поймано отмен, %",
}
SMALL_MONEY = 100_000
WALL_KINDS = {
    "caught": "отменят, но предоплата уже взята",
    "touched": "приедут, но с них взяли предоплату",
    "missed": "отменят, а предоплаты нет",
    "reliable": "приедут, предоплату не брали",
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


def hundred(counts: dict[str, float]) -> dict[str, int]:
    """Разложить количества на 100 плиток методом наибольших остатков: сумма ровно 100, доли сохраняются.

    Сначала каждой группе — целая часть её доли от 100, оставшиеся плитки — группам с наибольшими остатками
    (при равных остатках — в порядке ключей). Пустой вход (сумма 0) — нули.
    """
    total = sum(counts.values())
    if total <= 0:
        return {key: 0 for key in counts}
    exact = {key: value * 100 / total for key, value in counts.items()}
    tiles = {key: int(np.floor(value)) for key, value in exact.items()}
    for key in sorted(exact, key=lambda k: exact[k] - tiles[k], reverse=True)[:100 - sum(tiles.values())]:
        tiles[key] += 1
    return tiles


def wall_counts(frame: pd.DataFrame, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> dict[str, int]:
    """«Стена из 100 плиток»: сколько из 100 типичных броней среза попадает в каждую из четырёх групп.

    Модель берёт предоплату при риске не ниже порога по формуле; группы — WALL_KINDS: предоплата взята и бронь
    отменят (поймана), взята и гость приедет (задет), не взята и отменят (пропущена), не взята и приедет.
    Считаются все брони среза. Пустой срез — пустой словарь (стену не показываем).
    """
    if frame.empty:
        return {}
    prepay = frame["proba"].to_numpy(dtype=float) >= model_threshold(assumptions)
    canceled = frame["is_canceled"].to_numpy() == 1
    return hundred({"caught": float((prepay & canceled).sum()), "touched": float((prepay & ~canceled).sum()),
                    "missed": float((~prepay & canceled).sum()), "reliable": float((~prepay & ~canceled).sum())})


def wall_html(counts: dict[str, int], hotel: str = ALL_HOTELS) -> str:
    """HTML стены: сетка 10×10 плиток по группам из wall_counts, отметка отеля его цветом и легенда с числами.

    Для экранных читалок у сетки есть role="img" и подпись с теми же числами, что в легенде.
    """
    names = {"City Hotel": "городской отель", "Resort Hotel": "курортный отель"}
    shown = list(names) if hotel == ALL_HOTELS else [hotel]
    marks = "".join(f'<span class="hotel-mark" style="background:{HOTEL_COLORS[name]}"></span>' for name in shown)
    where = "оба отеля" if hotel == ALL_HOTELS else names[hotel]
    label = "Из 100 типичных броней: " + "; ".join(f"{counts[kind]} {text}" for kind, text in WALL_KINDS.items())
    tiles = "".join(f'<span class="tile tile-{kind}"></span>' * counts[kind] for kind in WALL_KINDS)
    legend = "".join(f'<li><span class="tile tile-{kind}"></span><b>{counts[kind]}</b><span>{text}</span></li>'
                     for kind, text in WALL_KINDS.items())
    return (f'<div class="wall-block"><p class="wall-title">Из 100 типичных броней: {marks}{where}</p>'
            f'<div class="wall" role="img" aria-label="{label}" title="{label}">{tiles}</div>'
            f'<ul class="wall-legend">{legend}</ul></div>')


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
    """Сумма с десятичной запятой: от 0,1 млн € — в миллионах (2,11 млн €), меньше — в тысячах (23,4 тыс. €)."""
    if abs(value) < SMALL_MONEY:
        return f"{comma(value / 1e3, 1)} тыс. €"
    return f"{comma(value / 1e6)} млн €"


def money_unit(values: pd.Series) -> tuple[float, str, int]:
    """Единица для ряда сумм: тысячи евро, если все суммы меньше 0,1 млн €, иначе миллионы.

    Возвращает делитель, подпись единицы и число знаков после запятой.
    """
    if values.abs().max() < SMALL_MONEY:
        return 1e3, "тыс. €", 1
    return 1e6, "млн €", 3


def saving_card(saving: float) -> tuple[str, str]:
    """Подпись и значение карточки «модель против предоплаты со всех»: дешевле, дороже или почти одинаково."""
    if abs(saving) < 50:
        return "Модель против «со всех»", "почти одинаково"
    if saving > 0:
        return "Модель дешевле «со всех» на", money(saving)
    return "Модель дороже «со всех» на", money(-saving)


def count(value: float) -> str:
    """Целое число с пробелом между тысячами: 4 898."""
    return f"{value:,.0f}".replace(",", " ")


def percent(share: float) -> str:
    """Доля в процентах без знаков после запятой: 60 %."""
    return f"{share * 100:.0f} %"


def comma(value: float, digits: int = 2) -> str:
    """Число с десятичной запятой: 0,17 (без «-0,00» у малых отрицательных)."""
    return f"{round(value, digits) + 0.0:.{digits}f}".replace(".", ",")


def lead_total_at_share(curves: pd.DataFrame, share: float) -> float | None:
    """Потери при выборе броней по сроку на заданной доле; None, если такой доли нет на кривой."""
    row = curves[(curves["method"] == "срок брони") & curves["share"].round(2).eq(round(share, 2))]
    return None if row.empty else float(row["total"].iloc[0])


def top_share_row(table: pd.DataFrame) -> pd.Series:
    """Строка правила «модель, топ-X %» из таблицы rules_table."""
    return table[table["policy"].str.startswith("модель, топ")].iloc[0]


def display_rules(table: pd.DataFrame) -> pd.DataFrame:
    """Таблица правил для показа: деньги в млн € (у малых срезов — в тыс. €) и доли в % строками с десятичной
    запятой, русские заголовки."""
    shown = table[list(RULE_COLUMNS)].copy()
    scale, unit, digits = money_unit(shown["total"])
    for column in ("total", "saving"):
        shown[column] = shown[column].map(lambda value: comma(value / scale, digits))
    for column in ("prepay_share", "touched_reliable", "caught_cancels"):
        shown[column] = shown[column].map(percent)
    return shown.rename(columns={key: name.format(unit=unit) for key, name in RULE_COLUMNS.items()})


def fig_share_curves(curves: pd.DataFrame, share: float) -> go.Figure:
    """Кривые «доля броней с предоплатой → потери» для модели и срока брони и отметка выбранной доли."""
    fig = go.Figure()
    divisor, unit, _ = money_unit(curves["total"])
    for method, dash, color in (("модель", "solid", COBALT), ("срок брони", "dash", STONE)):
        line = curves[curves["method"] == method]
        fig.add_trace(go.Scatter(
            x=line["share"] * 100, y=line["total"] / divisor, mode="lines+markers", name=f"брони выбирает {method}",
            line={"dash": dash, "color": color, "width": 2.5}, marker={"size": 6, "color": color},
            customdata=line["touched_reliable"] * 100,
            hovertemplate=f"предоплата с %{{x:.0f}} % броней<br>потери %{{y:.2f}} {unit}"
                          "<br>задето надёжных %{customdata:.0f} %<extra></extra>"))
    fig.add_vline(x=share * 100, line={"color": INK, "dash": "dot", "width": 1},
                  annotation_text=f"выбрано: {percent(share)}")
    fig.update_layout(**LAYOUT, xaxis_title="доля броней с предоплатой, %", yaxis_title=f"потери, {unit}",
                      legend={"orientation": "h", "y": -0.25})
    return fig


def fig_weeks(dates: pd.DataFrame, hotel: str) -> go.Figure:
    """По неделям заезда для одного отеля: столбики фактических отмен, линии ожидания модели и бенчмарка."""
    fig = go.Figure(week_traces(weekly_table(dates, hotel)))
    fig.update_traces(marker_color=OCHRE, opacity=0.75, selector={"type": "bar"})
    fig.update_traces(line_color=COBALT, marker_color=COBALT, selector={"name": "сколько ждала модель"})
    fig.update_traces(line_color=STONE, marker_color=STONE, selector={"mode": "lines+markers", "line.dash": "dash"})
    fig.update_layout(**LAYOUT, xaxis_title="неделя заезда", yaxis_title="отмен за неделю", xaxis_tickangle=-45,
                      legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"})
    return fig


def fig_heatmap(pivot: pd.DataFrame, counts: pd.DataFrame, title: str, color: str = COBALT) -> go.Figure:
    """Тепловая карта «сегмент × месяц» в процентах, от белого к color; в подсказке — число броней в клетке.

    Клетка без броней остаётся пустой: без текста и без подсказки.
    """
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=list(pivot.columns), y=list(pivot.index), colorscale=one_color_scale(color),
        zmin=0, zmax=100,
        text=pivot.map(lambda value: "" if pd.isna(value) else f"{value:.0f}").to_numpy(), texttemplate="%{text}",
        customdata=counts.fillna(0).to_numpy(), hoverongaps=False,
        hovertemplate="%{y}, %{x}: %{z:.0f} %<br>броней: %{customdata:.0f}<extra></extra>",
        colorbar={"title": "%"}))
    fig.update_layout(**LAYOUT, title=title, xaxis_title="месяц заезда",
                      yaxis_title={"text": "сегмент", "standoff": 12})
    return fig
