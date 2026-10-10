"""Оформление дашборда «азулежу»: цвета по смыслу, шрифт и общий вид графиков Plotly.

Цвет везде значит одно и то же: кобальт — городской отель и модель, бирюза — курортный отель, охра — отмены и
потери, серый камень — бенчмарки и простые правила. HTML-отчёт шага 4 оформлен отдельно и эти цвета не использует.
"""
import plotly.graph_objects as go

TILE = "#F6F8FB"
INK = "#1B2433"
COBALT = "#1F4E9C"
TEAL = "#2A8C8C"
OCHRE = "#C8962E"
STONE = "#8A94A6"
GRID = "#DDE3EC"
FONT = "Golos Text, sans-serif"
HOTEL_COLORS = {"City Hotel": COBALT, "Resort Hotel": TEAL}

TEMPLATE = go.layout.Template(layout={
    "font": {"family": FONT, "color": INK, "size": 13},
    "paper_bgcolor": TILE, "plot_bgcolor": "#FFFFFF",
    "colorway": [COBALT, STONE, OCHRE, TEAL],
    "xaxis": {"gridcolor": GRID, "linecolor": STONE, "zerolinecolor": GRID, "ticks": "", "automargin": True},
    "yaxis": {"gridcolor": GRID, "linecolor": STONE, "zerolinecolor": GRID, "ticks": "", "automargin": True},
    "hoverlabel": {"font": {"family": FONT}, "bgcolor": "white", "bordercolor": GRID},
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
    "margin": {"l": 60, "r": 20, "t": 40, "b": 40},
})
# Фон и шрифт дублируются вне шаблона: Streamlit подставляет свои значения, если их нет в самом layout.
LAYOUT = {"template": TEMPLATE, "separators": ", ", "paper_bgcolor": TILE, "plot_bgcolor": "#FFFFFF",
          "font": {"family": FONT, "color": INK, "size": 13}}


def one_color_scale(color: str) -> list[list]:
    """Однотонная шкала тепловой карты: от белого к цвету смысла (кобальт — модель, охра — отмены)."""
    return [[0.0, "#FFFFFF"], [1.0, color]]


CSS = f"""
[data-testid="stMainBlockContainer"] {{ padding-top: 2.5rem; }}
[data-testid="stHeading"] h1 {{ font-weight: 650; letter-spacing: -0.015em; }}
[data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {{ font-weight: 600; letter-spacing: -0.01em; }}
[data-testid="stMetricLabel"] p {{ white-space: normal; }}
[data-testid="stTabs"] {{ margin-top: 1.5rem; }}
[data-testid="stMetricValue"], .wall-legend b {{ font-variant-numeric: tabular-nums; }}
[data-testid="stMetricValue"] {{ font-weight: 600; }}
[class*="st-key-kpi_"] {{ border-left: 3px solid {STONE}; padding-left: 0.85rem; }}
.st-key-kpi_baseline {{ border-left-color: {OCHRE}; }}
.st-key-kpi_everyone {{ border-left-color: {STONE}; }}
.st-key-kpi_model {{ border-left-color: {COBALT}; }}
.st-key-kpi_touched {{ border-left-color: {INK}; }}
.wall-title {{ margin: 0 0 0.6rem; font-weight: 600; }}
.hotel-mark {{ display: inline-block; width: 0.7em; height: 0.7em; margin-right: 0.3em; border-radius: 2px; }}
.hotel-mark:last-of-type {{ margin-right: 0.45em; }}
.wall {{ display: grid; grid-template-columns: repeat(10, 26px); gap: 5px; width: max-content; }}
.tile {{ display: inline-block; box-sizing: border-box; width: 26px; height: 26px; border-radius: 3px;
         position: relative; }}
.tile::after {{ content: ""; position: absolute; left: 50%; top: 50%; width: 6px; height: 6px;
                transform: translate(-50%, -50%) rotate(45deg); }}
.tile-reliable {{ background: #E1E7F0; }}
.tile-reliable::after {{ background: #C9D3E2; }}
.tile-missed {{ background: {OCHRE}; }}
.tile-missed::after, .tile-caught::after {{ background: #E3C27A; }}
.tile-touched {{ background: #FFFFFF; border: 3px solid {COBALT}; }}
.tile-touched::after {{ background: #9DB2D6; }}
.tile-caught {{ background: {OCHRE}; border: 3px solid {COBALT}; }}
.wall-legend {{ list-style: none; margin: 1rem 0 0; padding: 0; }}
.wall-legend li {{ display: grid; grid-template-columns: 26px 2.2rem 1fr; gap: 0.6rem; align-items: center;
                   margin: 0 0 0.35rem; font-size: 0.95rem; }}
.wall-legend b {{ text-align: right; font-weight: 600; }}
"""
