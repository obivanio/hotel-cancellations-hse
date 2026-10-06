"""Интерактивный HTML-отчёт для revenue-менеджера: графики Plotly по таблицам reports/*.csv."""
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.evaluation import DEFAULT_ASSUMPTIONS
from src.overbooking import OVERBOOK_CONFIDENCE, OVERBOOKING_PATH, outcome_table
from src.policies import ALL_HOTELS, BASELINE, CURVES_PATH, POLICIES_PATH, REPORTS_DIR

REPORT_PATH = REPORTS_DIR / "report.html"
RISK_PATH = REPORTS_DIR / "risk_segments.csv"
LAYOUT = {"template": "plotly_white", "separators": ", "}
MONTH_NAMES = {1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "май", 6: "июн", 7: "июл", 8: "авг", 9: "сен", 10: "окт",
               11: "ноя", 12: "дек"}
TOP_CATEGORIES = ["потери, % от «как сейчас»", "задето надёжных гостей, %", "поймано отмен, %"]
SECTIONS = {
    "curve": "1. Доля предоплаты и потери: модель против срока брони",
    "slider": "2. Топ-X % броней по риску",
    "overbooking": "3. Календарь овербукинга",
    "risk": "4. Риск отмены по сегментам и месяцам",
    "assumptions": "Допущения и ограничения",
}


def risk_table(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Данные для графика 4: средний риск модели и факт отмен по отелю, сегменту и месяцу заезда."""
    frame = df.assign(proba=proba, hotel=df["hotel"].astype(str))
    parts = [frame.assign(hotel=ALL_HOTELS), frame]
    grouped = pd.concat(parts).groupby(["hotel", "market_segment", "arrival_month"], observed=True)
    return grouped.agg(bookings=("proba", "size"), mean_proba=("proba", "mean"),
                       cancel_rate=("is_canceled", "mean")).reset_index()


def _mln(value: float) -> str:
    """Сумма в миллионах евро с десятичной запятой: 2,11 млн €."""
    return f"{value / 1e6:.2f} млн €".replace(".", ",")


def _pct(share: float) -> str:
    """Доля в процентах: 60 %."""
    return f"{share * 100:.0f} %"


def _hotels(table: pd.DataFrame) -> list[str]:
    """Отели в порядке показа: сначала «все отели»."""
    return [ALL_HOTELS] + sorted(set(table["hotel"]) - {ALL_HOTELS})


def _hotel_menu(hotels: list[str], traces_per_hotel: int) -> list[dict]:
    """Выпадающий список «отель»: показывает только трассы выбранного отеля."""
    buttons = []
    for i, hotel in enumerate(hotels):
        visible = [j // traces_per_hotel == i for j in range(len(hotels) * traces_per_hotel)]
        buttons.append({"label": hotel, "method": "update", "args": [{"visible": visible}]})
    return [{"buttons": buttons, "direction": "down", "x": 1.0, "xanchor": "right", "y": 1.15, "yanchor": "top"}]


def fig_share_curve(curves: pd.DataFrame, policies: pd.DataFrame) -> go.Figure:
    """График 1: доля броней с предоплатой → потери для модели и срока брони, точки правил; выбор отеля."""
    fig = go.Figure()
    hotels = _hotels(curves)
    for i, hotel in enumerate(hotels):
        for method, dash in (("модель", "solid"), ("срок брони", "dash")):
            curve = curves[(curves["hotel"] == hotel) & (curves["method"] == method)]
            fig.add_trace(go.Scatter(
                x=curve["prepay_share"] * 100, y=curve["total"] / 1e6, mode="lines", name=method,
                line={"dash": dash}, visible=i == 0, customdata=curve["touched_reliable"] * 100,
                hovertemplate="предоплата с %{x:.0f} % броней<br>потери %{y:.2f} млн €"
                              "<br>задето надёжных %{customdata:.0f} %<extra></extra>"))
        points = policies[(policies["hotel"] == hotel) & ~policies["policy"].str.contains("топ")]
        fig.add_trace(go.Scatter(
            x=points["prepay_share"] * 100, y=points["total"] / 1e6, mode="markers+text", name="правила",
            text=points["policy"], textposition="top center", marker={"size": 10, "color": "black"},
            visible=i == 0, hovertemplate="%{text}<br>потери %{y:.2f} млн €<extra></extra>"))
    fig.update_layout(**LAYOUT, updatemenus=_hotel_menu(hotels, 3),
                      xaxis_title="доля броней с предоплатой, %", yaxis_title="потери, млн €")
    return fig


def fig_top_share(curves: pd.DataFrame) -> go.Figure:
    """График 2: ползунок «топ-X % по риску» — потери, задетые надёжные гости и пойманные отмены по отелям."""
    model = curves[curves["method"] == "модель"].set_index(["share", "hotel"])
    baseline = model.xs(0.0, level="share")["total"]
    shares = [share for share in model.index.unique("share") if share > 0]
    hotels = _hotels(curves)
    fig, steps = go.Figure(), []
    for k, share in enumerate(shares):
        rows = model.xs(share, level="share")
        for hotel in hotels:
            row = rows.loc[hotel]
            fig.add_trace(go.Bar(
                x=TOP_CATEGORIES, name=hotel, visible=k == 0, legendgroup=hotel,
                y=[row["total"] / baseline[hotel] * 100, row["touched_reliable"] * 100, row["caught_cancels"] * 100],
                hovertemplate="%{x}: %{y:.0f}<extra>" + hotel + "</extra>"))
        title = f"Предоплата с {_pct(share)} самых рискованных броней. Потери — " + "; ".join(
            f"{hotel}: {_mln(rows.loc[hotel, 'total'])}" for hotel in hotels)
        visible = [j // len(hotels) == k for j in range(len(shares) * len(hotels))]
        steps.append({"label": _pct(share), "method": "update", "args": [{"visible": visible}, {"title": title}]})
    fig.update_layout(**LAYOUT, barmode="group", yaxis_title="%", title=steps[0]["args"][1]["title"],
                      title_font_size=13, sliders=[{"steps": steps, "currentvalue": {"prefix": "топ-"}}])
    return fig


def fig_overbooking(dates: pd.DataFrame) -> go.Figure:
    """График 3: по датам теста — факт отмен, ожидание и осторожная оценка модели, ожидание бенчмарка."""
    fig = go.Figure()
    hotels = _hotels(dates)
    series = [("actual", "факт отмен", "markers", None), ("expected", "ожидание модели", "lines", "solid"),
              ("cautious", f"осторожная оценка ({_pct(OVERBOOK_CONFIDENCE)})", "lines", "dash"),
              ("bench_expected", "бенчмарк «месяц × день недели»", "lines", "dot")]
    for i, hotel in enumerate(hotels):
        part = dates[dates["hotel"] == hotel]
        for column, name, mode, dash in series:
            fig.add_trace(go.Scatter(
                x=part["arrival_date"], y=part[column], mode=mode, name=name, visible=i == 0,
                line={"dash": dash} if dash else None, hovertemplate="%{x|%d.%m.%Y}: %{y:.0f}<extra></extra>"))
    fig.update_layout(**LAYOUT, updatemenus=_hotel_menu(hotels, len(series)),
                      xaxis_title="дата заезда", xaxis_tickformat="%d.%m", yaxis_title="число отмен")
    return fig


def fig_risk(risk: pd.DataFrame) -> go.Figure:
    """График 4: средний риск отмены по месяцам заезда для каждого сегмента; выбор отеля, сегменты — легендой."""
    fig = go.Figure()
    hotels = _hotels(risk)
    segments = sorted(risk["market_segment"].unique())
    for i, hotel in enumerate(hotels):
        for segment in segments:
            part = risk[(risk["hotel"] == hotel) & (risk["market_segment"] == segment)]
            fig.add_trace(go.Scatter(
                x=part["arrival_month"].map(MONTH_NAMES), y=part["mean_proba"] * 100, mode="lines+markers",
                name=segment, visible=i == 0, customdata=np.c_[part["cancel_rate"] * 100, part["bookings"]],
                hovertemplate=segment + ", %{x}: риск %{y:.0f} %, факт %{customdata[0]:.0f} %"
                                        "<br>броней %{customdata[1]}<extra></extra>"))
    fig.update_layout(**LAYOUT, updatemenus=_hotel_menu(hotels, len(segments)),
                      xaxis_title="месяц заезда", yaxis_title="средний риск отмены, %")
    return fig


def _summary(policies: pd.DataFrame, curves: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """Числа для шапки и выводов (все отели): правила, сравнение при равной доле предоплаты, овербукинг."""
    total = policies[policies["hotel"] == ALL_HOTELS].set_index("policy")
    rule = total[total.index.str.startswith("модель, порог")].iloc[0]
    share = round(rule["prepay_share"] * 20) / 20
    at_share = curves[(curves["hotel"] == ALL_HOTELS) & np.isclose(curves["share"], share)].set_index("method")
    plans = outcomes[outcomes["walk_cost_share"] == 1.0].set_index("plan")
    return {"total": total, "rule": rule, "share": share, "at_share": at_share["total"], "plans": plans}


def _key_numbers(summary: dict) -> dict[str, str]:
    """Ключевые цифры для шапки отчёта."""
    rule, at_share, model = summary["rule"], summary["at_share"], summary["plans"].loc["модель"]
    return {
        "без предоплаты": _mln(summary["total"].loc[BASELINE, "total"]),
        rule.name: f"{_mln(rule['total'])} (задето {_pct(rule['touched_reliable'])} надёжных гостей)",
        f"предоплата с {_pct(summary['share'])} броней":
            f"модель {_mln(at_share['модель'])}, срок брони {_mln(at_share['срок брони'])}",
        "овербукинг по модели": f"перепродано {model['resold']:.0f} номеров, без номера {model['walked']:.0f} гостей",
    }


def _texts(summary: dict, dates: pd.DataFrame) -> dict[str, str]:
    """Короткие выводы под графиками; числа берутся из таблиц, а не пишутся руками."""
    total, at_share = summary["total"], summary["at_share"]
    best = total["total"].idxmin()
    pooled = dates[dates["hotel"] == ALL_HOTELS]
    model, bench = summary["plans"].loc["модель"], summary["plans"].loc["бенчмарк"]
    mean_expected = f"{pooled['expected'].mean():.1f}".replace(".", ",")
    mean_actual = f"{pooled['actual'].mean():.1f}".replace(".", ",")
    return {
        "curve": f"Меньше всего потерь у правила «{best}» — {_mln(total.loc[best, 'total'])}; «со всех» — "
                 f"{_mln(total.loc['со всех', 'total'])}, но задевает всех надёжных гостей. При предоплате с "
                 f"{_pct(summary['share'])} броней выбор по модели теряет {_mln(at_share['модель'])}, по сроку "
                 f"брони — {_mln(at_share['срок брони'])}.",
        "slider": "Двигайте ползунок: чем больше доля броней с предоплатой, тем больше поймано отмен и задето "
                  "надёжных гостей; потери сначала падают, потом растут. Отель скрывается щелчком по легенде.",
        "overbooking": f"В среднем на дату ожидалось {mean_expected} отмен, случилось {mean_actual}. Осторожная "
                       f"оценка модели перепродаёт {model['resold']:.0f} номеров (бенчмарк — "
                       f"{bench['resold']:.0f}), но в {model['short_dates']:.0f} парах «отель × дата» гостям не "
                       f"хватило бы номера ({model['walked']:.0f} гостей).",
        "risk": "Риск — средняя вероятность отмены по модели; в подсказке — фактическая доля отмен. Сегменты "
                "включаются и выключаются щелчком по легенде. Это связь, а не причина.",
    }


def _assumptions_html() -> str:
    """Блок «Допущения и ограничения»."""
    items = [
        f"Не перепродаётся {_pct(DEFAULT_ASSUMPTIONS.unsold_share)} стоимости отменённой брони; надёжный гость уходит "
        f"из-за требования предоплаты с вероятностью {_pct(DEFAULT_ASSUMPTIONS.loss_prob)} (чувствительность — в "
        "ноутбуке 04).",
        "Брони «без возврата» уже оплачены: их отмена ничего не стоит отелю.",
        f"Овербукинг: осторожная оценка — число отмен, которое наступит с вероятностью не ниже "
        f"{_pct(OVERBOOK_CONFIDENCE)}; «гость без номера» стоит одну стоимость его брони (проверено 0,5 и 2).",
        "Вместимости отелей в данных нет: перепродажа считается сверх фактически забронированного.",
        "Отмены считаются независимыми; групповые отмены это нарушают.",
        "Тест — май–август 2017 года, два отеля в Португалии; связь, а не причина.",
        "Отчёт статичный: фильтры заготовлены для каждого графика отдельно, комбинация «отель × сегмент × месяц» "
        "не пересчитывается.",
    ]
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


STYLE = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; max-width: 1100px; margin: 0 auto;
       padding: 16px; color: #1f2933; background: #ffffff; line-height: 1.5; }
h1 { font-size: 1.6em; } h2 { font-size: 1.25em; margin-top: 2em; }
.keys { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.key { background: #f3f5f8; border-radius: 8px; padding: 10px 14px; }
.key b { display: block; font-size: 0.85em; color: #52606d; }
nav ul { padding-left: 1.2em; list-style: none; }
"""


def _read_tables(reports_dir: Path) -> dict[str, pd.DataFrame]:
    """Таблицы шага 4 из CSV: правила, кривые долей предоплаты, даты овербукинга, риск по сегментам."""
    return {
        "policies": pd.read_csv(reports_dir / POLICIES_PATH.name),
        "curves": pd.read_csv(reports_dir / CURVES_PATH.name),
        "dates": pd.read_csv(reports_dir / OVERBOOKING_PATH.name, parse_dates=["arrival_date"]),
        "risk": pd.read_csv(reports_dir / RISK_PATH.name),
    }


def build_report(reports_dir: Path = REPORTS_DIR, path: Path = REPORT_PATH) -> Path:
    """Собрать самодостаточный reports/report.html из CSV шага 4 (plotly.js встроен, интернет не нужен)."""
    tables = _read_tables(reports_dir)
    summary = _summary(tables["policies"], tables["curves"], outcome_table(tables["dates"]))
    texts = _texts(summary, tables["dates"])
    figures = {
        "curve": fig_share_curve(tables["curves"], tables["policies"]), "slider": fig_top_share(tables["curves"]),
        "overbooking": fig_overbooking(tables["dates"]), "risk": fig_risk(tables["risk"]),
    }
    keys = "".join(f'<div class="key"><b>{escape(name)}</b>{escape(value)}</div>'
                   for name, value in _key_numbers(summary).items())
    toc = "".join(f'<li><a href="#{key}">{escape(title)}</a></li>' for key, title in SECTIONS.items())
    body = []
    for i, (key, fig) in enumerate(figures.items()):
        chart = fig.to_html(full_html=False, include_plotlyjs=i == 0, config={"displaylogo": False})
        body.append(f'<section id="{key}"><h2>{escape(SECTIONS[key])}</h2><p>{escape(texts[key])}</p>{chart}</section>')
    body.append(f'<section id="assumptions"><h2>{SECTIONS["assumptions"]}</h2>{_assumptions_html()}</section>')
    html = (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Предоплата и овербукинг</title><style>{STYLE}</style></head><body>"
        "<h1>Отмены броней: с кого брать предоплату и насколько овербукировать</h1>"
        "<p>Тест: брони с заездом в мае–августе 2017 года, два отеля. Модель — CatBoost с калибровкой (шаг 3).</p>"
        f'<div class="keys">{keys}</div><nav><h2>Оглавление</h2><ul>{toc}</ul></nav>{"".join(body)}</body></html>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
