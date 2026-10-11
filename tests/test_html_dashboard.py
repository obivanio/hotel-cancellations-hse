"""Тесты сборки дашборда reports/dashboard.html."""
import json
import re

import pandas as pd
import pytest

from src.dashboard import SCORED_TEST_PATH
from src.html_dashboard import MAX_BYTES, build_dashboard, pack_bookings

SECTIONS = ["С кого брать предоплату, чтобы терять меньше", "Модель против правила «по сроку брони»",
            "Сколько номеров продать сверх брони", "Где отменяют чаще", "Как это устроено", "Что делать бизнесу",
            "Допущения", "В этом срезе нет броней. Выберите другой сегмент или месяц."]


@pytest.fixture(name="page", scope="module")
def fixture_page(tmp_path_factory):
    """Дашборд, собранный во временную папку, и его текст."""
    path = build_dashboard(tmp_path_factory.mktemp("dashboard") / "dashboard.html")
    return path, path.read_text(encoding="utf-8")


def test_dashboard_has_sections_and_data(page):
    """Все разделы на месте, шаблон заполнен, данные — все 17 351 бронь теста и цифры рекомендаций."""
    _, html = page
    assert all(section in html for section in SECTIONS)
    assert not re.search(r"\{\{(fonts|css|data|js|plotly)\}\}", html)
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))
    assert len(data["bookings"]["h"]) == 17351
    assert data["verdict"]["covered_city"] == pytest.approx(0.7273, abs=1e-4)
    assert data["verdict"]["covered_all"] == pytest.approx(0.851, abs=1e-3)
    assert data["obTotals"]["все отели"]["resold"] == 4898
    assert set(data["weeks"]) == {"все отели", "City Hotel", "Resort Hotel"}
    assert set(data["figures"]) == {"queue", "curves", "weeks", "heatmap", "segments"}
    assert 'src="figures/20_model_full_pipeline.html?theme=light"' in html


def test_dashboard_offline_and_small(page):
    """Plotly.js встроен один раз; вне библиотеки — ни одного обращения к сети; шрифты встроены; файл ≤ 7 МБ."""
    path, html = page
    assert html.count('<script id="plotly-js">') == 1
    own = re.sub(r'<script id="plotly-js">.*?</script>', "", html, flags=re.S)
    assert not re.search(r"""(src|href)\s*=\s*["']?\s*(https?:)?//""", own)
    assert not re.search(r"""url\(\s*["']?\s*(https?:)?//""", own)
    assert "@import" not in own
    assert own.count("url(data:font/ttf;base64,") == 2
    assert path.stat().st_size <= MAX_BYTES == 7_000_000


def test_pack_bookings_round_trip():
    """Упакованные столбцы восстанавливают исходные брони без потерь и в том же порядке."""
    frame = pd.read_csv(SCORED_TEST_PATH).head(500)
    packed = pack_bookings(frame)
    assert [packed["hotels"][i] for i in packed["h"]] == frame["hotel"].tolist()
    assert [packed["segments"][i] for i in packed["s"]] == frame["market_segment"].tolist()
    assert [packed["probas"][i] for i in packed["p"]] == frame["proba"].tolist()
    assert packed["value"] == frame["value"].tolist()
