"""Сборка дашборда reports/dashboard.html: один файл с данными, оформлением, логикой, шрифтами
и библиотекой Plotly.js, без сети.

Запуск: python -m src.html_dashboard
"""
import base64
import json

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from src.dashboard import (
    HOTEL_NAMES, SCORED_TEST_PATH, SEGMENT_NAMES, overbooking_totals, verdict_numbers, weekly_overbooking,
)
from src.dashboard_figures import CONFIG, figure_specs
from src.evaluation import DEFAULT_ASSUMPTIONS
from src.overbooking import OVERBOOKING_PATH
from src.pipeline import ROOT
from src.policies import POLICIES_PATH, REPORTS_DIR

ASSETS_DIR = ROOT / "src" / "dashboard_assets"
FONTS_DIR = ROOT / "static" / "fonts"
DASHBOARD_PATH = REPORTS_DIR / "dashboard.html"
FONTS = {"Golos Text": ("GolosText-Variable.ttf", "400 900"), "Unbounded": ("Unbounded-Variable.ttf", "200 900")}
MONTH_FULL = {5: "май", 6: "июнь", 7: "июль", 8: "август"}
MONTH_SHORT = {5: "май", 6: "июн", 7: "июл", 8: "авг"}
MAX_BYTES = 7_000_000


def pack_bookings(frame: pd.DataFrame) -> dict:
    """Брони столбцами без повторения имён полей: отель и сегмент — номера в списках, риск — номер в списке значений.

    Порядок строк сохраняется: от него зависит порядок броней с одинаковым риском в очереди.
    """
    hotels = sorted(frame["hotel"].unique())
    segments = sorted(frame["market_segment"].unique())
    probas, proba_index = np.unique(frame["proba"].to_numpy(dtype=float), return_inverse=True)
    return {
        "hotels": hotels, "segments": segments,
        "h": frame["hotel"].map(hotels.index).tolist(), "s": frame["market_segment"].map(segments.index).tolist(),
        "m": frame["arrival_month"].astype(int).tolist(), "lead": frame["lead_time"].astype(float).tolist(),
        "value": frame["value"].astype(float).tolist(), "cancel": frame["is_canceled"].astype(int).tolist(),
        "probas": probas.tolist(), "p": proba_index.astype(int).tolist(),
    }


def page_data(frame: pd.DataFrame, dates: pd.DataFrame, policies: pd.DataFrame) -> dict:
    """Всё, что нужно странице: брони теста, подписи, овербукинг по отелям и цифры рекомендаций."""
    order = frame.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False).index.tolist()
    weeks = {}
    for hotel in HOTEL_NAMES:
        table = weekly_overbooking(dates, hotel)
        weeks[hotel] = {"label": table["label"].tolist(), "month": table["month"].astype(int).tolist(),
                        "actual": table["actual"].astype(int).tolist(),
                        "cautious": table["cautious"].astype(int).tolist(),
                        "bench": table["bench_cautious"].astype(int).tolist()}
    return {
        "bookings": pack_bookings(frame), "segmentOrder": order,
        "segmentNames": {code: SEGMENT_NAMES.get(code, code) for code in order},
        "months": sorted(MONTH_FULL), "monthFull": MONTH_FULL, "monthShort": MONTH_SHORT,
        "weeks": weeks,
        "obTotals": {hotel: {key: float(value) for key, value in overbooking_totals(dates, hotel).items()}
                     for hotel in HOTEL_NAMES},
        "verdict": {key: float(value) for key, value in verdict_numbers(policies, dates).items()},
        "figures": figure_specs(frame, dates), "plotlyConfig": CONFIG,
        "defaults": {"unsold": DEFAULT_ASSUMPTIONS.unsold_share, "loss": DEFAULT_ASSUMPTIONS.loss_prob},
    }


def font_faces() -> str:
    """CSS @font-face со шрифтами, встроенными в base64 (лицензия OFL, файлы — static/fonts)."""
    faces = []
    for family, (file_name, weights) in FONTS.items():
        encoded = base64.b64encode((FONTS_DIR / file_name).read_bytes()).decode("ascii")
        faces.append(f'@font-face {{ font-family: "{family}"; font-weight: {weights}; font-display: swap; '
                     f'src: url(data:font/ttf;base64,{encoded}) format("truetype"); }}')
    return "\n".join(faces)


def render_page(data: dict) -> str:
    """Шаблон dashboard.html с подставленными шрифтами, оформлением, данными и логикой."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (ASSETS_DIR / "dashboard.html").read_text(encoding="utf-8")
    for key, value in (("{{fonts}}", font_faces()), ("{{css}}", (ASSETS_DIR / "dashboard.css").read_text("utf-8")),
                       ("{{data}}", payload), ("{{js}}", (ASSETS_DIR / "dashboard.js").read_text("utf-8"))):
        page = page.replace(key, value)
    page = page.replace("{{plotly}}", get_plotlyjs())
    return page


def build_dashboard(path=DASHBOARD_PATH):
    """Собрать reports/dashboard.html из reports/scored_test.csv, overbooking.csv и policies.csv; вернуть путь.

    Файл больше MAX_BYTES — ошибка: дашборд должен легко пересылаться.
    """
    data = page_data(pd.read_csv(SCORED_TEST_PATH), pd.read_csv(OVERBOOKING_PATH), pd.read_csv(POLICIES_PATH))
    page = render_page(data)
    size = len(page.encode("utf-8"))
    if size > MAX_BYTES:
        raise ValueError(f"дашборд весит {size} байт, больше {MAX_BYTES}")
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    built = build_dashboard()
    print(f"{built.relative_to(ROOT)}: {built.stat().st_size / 1e6:.1f} МБ")
