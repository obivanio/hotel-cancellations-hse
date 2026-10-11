"""Графики дашборда на Plotly: вид каждого графика задаёт Python, браузер подставляет пересчитанные данные.

Каждая функция строит go.Figure на срезе по умолчанию (оба отеля, все сегменты, май–август, допущения 50 % / 10 %).
Номера трасс и элементов layout — договор со страницей: dashboard.js меняет данные трасс по номеру.
Числа передаются списками (.tolist()), иначе fig.to_json() кодирует их в bdata.
"""
import json

import pandas as pd
import plotly.graph_objects as go

from src.dashboard import (
    BUCKETS, SEGMENT_NAMES, key_numbers, queue_buckets, risk_tables, segment_table, share_curves, weekly_overbooking,
)
from src.policies import ALL_HOTELS

INK, INK_2, COBALT, SAFFRON = "#0E2350", "#4A5878", "#1D3FBF", "#E8A317"
MIST, RULE, STONE, GRID, HEAT_LOW = "#C9D3EA", "#DCE2EC", "#6B7894", "#EEF1F6", "#FFF6E0"
FONT = "Golos Text, system-ui, sans-serif"
OFF_OPACITY = 0.45
CONFIG = {"displaylogo": False, "responsive": True, "locale": "ru", "scrollZoom": False,
          "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines", "sendChartToCloud"],
          "toImageButtonOptions": {"scale": 2}}


def style(fig: go.Figure, height: int, legend: bool = False) -> go.Figure:
    """Общий вид: шрифт Golos Text, палитра дашборда, десятичная запятая, белый фон, тонкая сетка;
    подсказки — тёмный текст на белом (читается над любым цветом трассы)."""
    fig.update_layout(template="none", height=height, margin={"l": 56, "r": 16, "t": 12, "b": 44},
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", separators=", ",
                      font={"family": FONT, "size": 13, "color": INK_2},
                      hoverlabel={"font": {"family": FONT, "color": INK}, "bgcolor": "#FFFFFF", "bordercolor": RULE},
                      showlegend=legend, legend={"orientation": "h", "x": 0, "y": 1.16})
    fig.update_xaxes(showgrid=False, linecolor=RULE, ticks="", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=RULE, ticks="", zeroline=False)
    return fig


def queue_figure(frame: pd.DataFrame, pct: int) -> go.Figure:
    """Очередь броней: 100 столбиков по риску, шафран — доля отмен (сглажена внутри групп одинакового риска);
    черта «предоплата с pct %»; столбики правее черты бледнее."""
    buckets = queue_buckets(frame)
    x = list(range(1, BUCKETS + 1))
    share = (buckets["cancel_share"] * 100).round(4).tolist()
    custom = [[int(n), round(float(r) * 100, 2)] for n, r in zip(buckets["bookings"], buckets["mean_risk"])]
    opacity = [1.0 if n < pct else OFF_OPACITY for n in range(BUCKETS)]
    fig = go.Figure()
    fig.add_bar(x=x, y=share, name="отменили", marker={"color": SAFFRON, "opacity": opacity}, customdata=custom,
                hovertemplate="%{x}-й процент очереди<br>броней: %{customdata[0]}<br>доля отмен: %{y:.0f} %"
                              "<br>средний риск: %{customdata[1]:.0f} %<extra></extra>")
    fig.add_bar(x=x, y=[100 - value for value in share], name="приехали",
                marker={"color": MIST, "opacity": opacity}, hoverinfo="skip")
    fig.update_layout(
        barmode="stack", bargap=0.18,
        shapes=[{"type": "line", "x0": pct + 0.5, "x1": pct + 0.5, "y0": 0, "y1": 1, "yref": "paper",
                 "line": {"color": INK, "width": 3}}],
        annotations=[{"x": pct + 0.5, "y": 1.12, "yref": "paper", "text": f"предоплата с {pct} %",
                      "showarrow": False, "font": {"color": "#FFFFFF", "size": 13}, "bgcolor": INK, "borderpad": 5}])
    style(fig, 260)
    fig.update_layout(margin={"l": 56, "r": 16, "t": 40, "b": 44})
    fig.update_xaxes(range=[0.4, BUCKETS + 0.6], showticklabels=False,
                     title_text="брони по риску отмены: слева самые рискованные, каждый столбик — 1 % броней")
    fig.update_yaxes(range=[0, 100], ticksuffix=" %")
    return fig


def curves_figure(frame: pd.DataFrame, pct: int) -> go.Figure:
    """Потери от доли броней с предоплатой: модель против правила по сроку брони; точка — выбор на черте."""
    curves = share_curves(frame)
    lead = curves[curves["method"] == "срок брони"]
    model = curves[curves["method"] == "модель"]
    model_y = (model["total"] / 1e6).round(6).tolist()
    lead_y = (lead["total"] / 1e6).round(6).tolist()
    fig = go.Figure()
    fig.add_scatter(x=lead["pct"].tolist(), y=lead_y, name="правило по сроку брони",
                    line={"color": STONE, "width": 2.5, "dash": "dash"},
                    hovertemplate="%{x} броней: %{y:.2f} млн €<extra>по сроку брони</extra>")
    fig.add_scatter(x=model["pct"].tolist(), y=model_y, name="модель", line={"color": COBALT, "width": 3.5},
                    hovertemplate="%{x} броней: %{y:.2f} млн €<extra>модель</extra>")
    fig.add_scatter(x=[pct], y=[model_y[pct]], mode="markers", marker={"color": COBALT, "size": 12},
                    showlegend=False, hoverinfo="skip")
    saving = lead_y[pct] - model_y[pct]
    fig.update_layout(annotations=[{"x": pct, "y": model_y[pct], "text": f"модель экономит {saving:.2f} млн €"
                                    .replace(".", ","), "showarrow": False, "yshift": -22,
                                    "font": {"color": "#8F5F00", "size": 13}, "bgcolor": "rgba(255,255,255,0.85)",
                                    "borderpad": 2}])
    style(fig, 320, legend=True)
    fig.update_xaxes(range=[0, 100], ticksuffix=" %", title_text="доля броней с предоплатой")
    fig.update_yaxes(rangemode="tozero", title_text="потери, млн €")
    return fig


def weeks_figure(dates: pd.DataFrame, hotel: str = ALL_HOTELS) -> go.Figure:
    """Овербукинг по неделям заезда: отмены на деле, сколько продаём сверх брони с моделью и без неё."""
    weeks = weekly_overbooking(dates, hotel)
    labels = weeks["label"].tolist()
    fig = go.Figure()
    fig.add_bar(x=labels, y=weeks["actual"].astype(int).tolist(), name="отменили на деле", marker_color=MIST)
    fig.add_scatter(x=labels, y=weeks["cautious"].astype(int).tolist(), name="продаём сверх брони (модель)",
                    line={"color": COBALT, "width": 3.5})
    fig.add_scatter(x=labels, y=weeks["bench_cautious"].astype(int).tolist(), name="без модели",
                    line={"color": STONE, "width": 2.5, "dash": "dash"})
    fig.update_traces(hovertemplate="%{y}")
    style(fig, 320, legend=True)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(nticks=6, tickangle=0)
    fig.update_yaxes(rangemode="tozero", title_text="номеров за неделю")
    return fig


def heatmap_figure(frame: pd.DataFrame, order: list[str]) -> go.Figure:
    """Доля отмен «сегмент × месяц» в %: строки в порядке order (по убыванию доли отмен), пустая клетка — прочерк;
    в подсказке — число броней и средний риск модели."""
    tables = risk_tables(frame)
    cancel = tables["cancel"].reindex(order)
    z = [[None if pd.isna(value) else round(float(value), 4) for value in row] for row in cancel.to_numpy()]
    text = [["–" if value is None else f"{value:.0f} %" for value in row] for row in z]
    bookings = tables["bookings"].reindex(index=order, columns=cancel.columns).fillna(0).to_numpy()
    risk = tables["risk"].reindex(index=order, columns=cancel.columns).to_numpy()
    custom = [[[f"{int(n):,}".replace(",", " "), None if pd.isna(r) else round(float(r), 2)]
               for n, r in zip(row_n, row_r)] for row_n, row_r in zip(bookings, risk)]
    fig = go.Figure(go.Heatmap(
        z=z, x=cancel.columns.tolist(), y=[SEGMENT_NAMES.get(code, code) for code in order], text=text,
        customdata=custom, texttemplate="%{text}", textfont={"color": INK, "size": 13},
        colorscale=[[0, HEAT_LOW], [1, SAFFRON]], zmin=0, zmax=60, showscale=False, xgap=4, ygap=4, hoverongaps=False,
        hovertemplate="<b>%{y}, %{x}</b><br>броней: %{customdata[0]}<br>отменили: %{text}"
                      "<br>средний риск по модели: %{customdata[1]:.0f} %<extra></extra>"))
    style(fig, 320)
    fig.update_layout(margin={"l": 140, "r": 16, "t": 36, "b": 12})
    fig.update_yaxes(autorange="reversed", showgrid=False, linecolor="#FFFFFF")
    fig.update_xaxes(side="top", linecolor="#FFFFFF")
    return fig


def segments_figure(frame: pd.DataFrame) -> go.Figure:
    """Потери по сегментам: без предоплаты и с предоплатой по порогу модели (горизонтальные столбики, млн €)."""
    table = segment_table(frame)
    names = [SEGMENT_NAMES.get(code, code) for code in table["segment"]]
    fig = go.Figure()
    fig.add_bar(y=names, x=(table["baseline"] / 1e6).round(6).tolist(), orientation="h", name="без предоплаты",
                marker_color=MIST, hovertemplate="%{y}: %{x:.2f} млн €<extra>без предоплаты</extra>")
    fig.add_bar(y=names, x=(table["model"] / 1e6).round(6).tolist(), orientation="h", name="с моделью",
                marker_color=COBALT, hovertemplate="%{y}: %{x:.2f} млн €<extra>с моделью</extra>")
    style(fig, 320, legend=True)
    fig.update_layout(barmode="group", bargap=0.25, margin={"l": 140, "r": 16, "t": 12, "b": 44})
    fig.update_xaxes(showgrid=True, gridcolor=GRID, title_text="потери, млн €", rangemode="tozero")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def figure_specs(frame: pd.DataFrame, dates: pd.DataFrame) -> dict[str, dict]:
    """JSON всех графиков для страницы: {"queue" | "curves" | "weeks" | "heatmap" | "segments": {data, layout}}."""
    pct = key_numbers(frame)["threshold_pct"]
    order = frame.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False).index.tolist()
    figures = {"queue": queue_figure(frame, pct), "curves": curves_figure(frame, pct), "weeks": weeks_figure(dates),
               "heatmap": heatmap_figure(frame, order), "segments": segments_figure(frame)}
    return {name: json.loads(fig.to_json()) for name, fig in figures.items()}
