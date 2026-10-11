"""Тесты сборки дашборда reports/dashboard.html."""
import pandas as pd

from src.dashboard import SCORED_TEST_PATH
from src.html_dashboard import pack_bookings


def test_pack_bookings_round_trip():
    """Упакованные столбцы восстанавливают исходные брони без потерь и в том же порядке."""
    frame = pd.read_csv(SCORED_TEST_PATH).head(500)
    packed = pack_bookings(frame)
    assert [packed["hotels"][i] for i in packed["h"]] == frame["hotel"].tolist()
    assert [packed["segments"][i] for i in packed["s"]] == frame["market_segment"].tolist()
    assert [packed["probas"][i] for i in packed["p"]] == frame["proba"].tolist()
    assert packed["value"] == frame["value"].tolist()
