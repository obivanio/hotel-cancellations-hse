"""Сверка расчётов браузера (dashboard.js) с эталонными функциями Python на бронях теста."""
import json
import shutil
import subprocess

import pandas as pd
import pytest

from src.dashboard import (
    SCORED_TEST_PATH, comma, count, filter_frame, key_numbers, money, percent, queue_buckets, queue_outcome,
    risk_tables, segment_table, share_curves,
)
from src.evaluation import Assumptions
from src.html_dashboard import ASSETS_DIR, pack_bookings
from src.policies import ALL_HOTELS

FRAME = pd.read_csv(SCORED_TEST_PATH)
SEGMENTS = sorted(FRAME["market_segment"].unique())
MONTHS = [5, 6, 7, 8]
SLICES = {
    "все": (ALL_HOTELS, SEGMENTS, MONTHS, 0.5, 0.1),
    "городской": ("City Hotel", SEGMENTS, MONTHS, 0.5, 0.1),
    "курортный": ("Resort Hotel", SEGMENTS, MONTHS, 0.5, 0.1),
    "группы": (ALL_HOTELS, ["Groups"], MONTHS, 0.5, 0.1),
    "июнь": (ALL_HOTELS, SEGMENTS, [6], 0.5, 0.1),
    "допущения 30 % и 20 %": (ALL_HOTELS, SEGMENTS, MONTHS, 0.3, 0.2),
}
FORMAT_VALUES = [0.0, -1_234_567.0, 40.0, 1_940.0, 23_400.0, 99_999.0, 100_000.0, 563_615.066, 644_810.346,
                 2_109_479.81]
RUNNER = """
const Calc = require(process.argv[1]);
const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
const data = input.bookings;
const out = { slices: {}, formats: {} };
for (const [name, s] of Object.entries(input.slices)) {
  const a = { unsold: s.unsold, loss: s.loss };
  const idx = Calc.filterIdx(data, s.hotel, s.segments, s.months);
  const key = Calc.keyNumbers(data, idx, a);
  const curves = Calc.shareCurves(data, idx, a);
  out.slices[name] = {
    key, segments: Calc.segmentTable(data, idx, a), at_threshold: Calc.queueOutcome(data, idx, key.threshold_pct, a),
    at_30: Calc.queueOutcome(data, idx, 30, a),
    buckets: Calc.queueBuckets(data, Calc.queueOrder(data, idx, "proba")),
    model: curves["модель"].map((r) => r.total), lead: curves["срок брони"].map((r) => r.total),
    touched: curves["модель"].map((r) => r.touched_share), heat: Calc.heatCells(data, idx),
  };
}
out.formats = input.formats.map((v) => [Calc.money(v), Calc.count(v), Calc.percent(v / 3e6), Calc.comma(v / 1e6)]);
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(name="browser", scope="module")
def fixture_browser():
    """Результаты dashboard.js на шести срезах: Node запускает чистые функции расчёта."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js не найден: сверка расчётов браузера пропущена")
    payload = {
        "bookings": pack_bookings(FRAME), "formats": FORMAT_VALUES,
        "slices": {name: {"hotel": hotel, "segments": segments, "months": months, "unsold": unsold, "loss": loss}
                   for name, (hotel, segments, months, unsold, loss) in SLICES.items()},
    }
    result = subprocess.run([node, "-e", RUNNER, str(ASSETS_DIR / "dashboard.js")], input=json.dumps(payload),
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def python_slice(name: str) -> tuple[pd.DataFrame, Assumptions]:
    """Срез броней и допущения из SLICES — то же, что получает браузер."""
    hotel, segments, months, unsold, loss = SLICES[name]
    return filter_frame(FRAME, hotel, segments, months), Assumptions(unsold, loss)


@pytest.mark.parametrize("name", list(SLICES))
def test_key_numbers_and_outcome_match(browser, name):
    """Опорные цифры и итог черты (на пороге и на 30 %) совпадают с Python."""
    frame, assumptions = python_slice(name)
    js = browser["slices"][name]
    numbers = key_numbers(frame, assumptions)
    assert js["key"] == pytest.approx({key: float(value) for key, value in numbers.items()}, rel=1e-9, abs=1e-6)
    for field, pct in (("at_threshold", numbers["threshold_pct"]), ("at_30", 30)):
        expected = queue_outcome(frame, pct, assumptions)
        assert js[field] == pytest.approx({key: float(value) for key, value in expected.items()}, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("name", list(SLICES))
def test_buckets_and_curves_match(browser, name):
    """100 полосок очереди и обе кривые «доля броней → потери» совпадают с Python."""
    frame, assumptions = python_slice(name)
    js = browser["slices"][name]
    assert pd.DataFrame(js["buckets"]).to_dict("list") == pytest.approx(
        queue_buckets(frame).astype(float).to_dict("list"), rel=1e-9, abs=1e-6)
    curves = share_curves(frame, assumptions)
    model, lead = curves[curves["method"] == "модель"], curves[curves["method"] == "срок брони"]
    assert js["model"] == pytest.approx(model["total"].tolist(), rel=1e-9, abs=1e-6)
    assert js["lead"] == pytest.approx(lead["total"].tolist(), rel=1e-9, abs=1e-6)
    assert js["touched"] == pytest.approx(model["touched_share"].tolist(), rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("name", list(SLICES))
def test_heatmap_match(browser, name):
    """Тепловая карта: в каждой непустой клетке «сегмент × месяц» — то же число броней, доля отмен и средний риск."""
    frame, _ = python_slice(name)
    tables = risk_tables(frame)
    months = dict(zip(tables["bookings"].columns, sorted(frame["arrival_month"].unique())))
    expected = {f"{seg}|{months[col]}": {"bookings": float(tables["bookings"].loc[seg, col]),
                                         "cancel": tables["cancel"].loc[seg, col], "risk": tables["risk"].loc[seg, col]}
                for seg in tables["bookings"].index for col in tables["bookings"].columns
                if tables["bookings"].loc[seg, col] > 0}
    heat = browser["slices"][name]["heat"]
    assert heat.keys() == expected.keys()
    for key, cell in expected.items():
        assert {field: heat[key][field] for field in cell} == pytest.approx(cell, rel=1e-9, abs=1e-6)


def test_formats_match(browser):
    """Подписи чисел в браузере такие же, как у форматтеров Python: деньги, целые, проценты, запятая."""
    expected = [[money(v), count(v), percent(v / 3e6), comma(v / 1e6)] for v in FORMAT_VALUES]
    assert browser["formats"] == expected


@pytest.mark.parametrize("name", list(SLICES))
def test_segment_table_match(browser, name):
    """Таблица по сегментам в браузере — те же строки, порядок и числа, что у Python."""
    frame, assumptions = python_slice(name)
    expected = segment_table(frame, assumptions)
    js = pd.DataFrame(browser["slices"][name]["segments"], columns=expected.columns)
    assert js["segment"].tolist() == expected["segment"].tolist()
    numeric = ["bookings", "cancel_share", "baseline", "model", "saving"]
    assert js[numeric].astype(float).to_dict("list") == pytest.approx(
        expected[numeric].astype(float).to_dict("list"), rel=1e-9, abs=1e-6)
