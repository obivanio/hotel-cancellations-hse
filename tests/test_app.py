"""Smoke-тесты дашборда: приложение запускается, фильтры пересчитывают цифры, пустой срез не ломает экран."""
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard import HOTELS
from src.pipeline import ROOT

APP_PATH = str(ROOT / "app.py")
TABS = ["Правила предоплаты", "Овербукинг", "Риск по сегментам", "Как собрана модель", "Допущения и ограничения"]


@pytest.fixture(name="app")
def fixture_app(tmp_path, monkeypatch):
    """Приложение, запущенное из чужой папки: пути к данным не должны зависеть от текущей папки."""
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def _baseline(app: AppTest) -> str:
    """Значение карточки «Без предоплаты»."""
    return next(metric.value for metric in app.metric if metric.label == "Без предоплаты")


def test_app_runs_with_sidebar_and_tabs(app):
    """Без ошибок; в боковой панели фильтры и ползунки допущений; все пять вкладок; карточки с десятичной запятой."""
    assert not app.exception
    assert [radio.key for radio in app.sidebar.radio] == ["hotel"]
    assert {box.key for box in app.sidebar.multiselect} == {"segments", "months"}
    assert {slider.key for slider in app.sidebar.slider} == {"unsold", "loss"}
    assert [tab.label for tab in app.tabs] == TABS
    assert _baseline(app) == "2,11 млн €"


def test_hotel_filter_changes_key_number(app):
    """Смена отеля пересчитывает ключевую цифру (кеш не отдаёт старое), возврат — возвращает её."""
    before = _baseline(app)
    app.sidebar.radio(key="hotel").set_value("городской").run()
    assert not app.exception
    assert _baseline(app) != before
    app.sidebar.radio(key="hotel").set_value(list(HOTELS)[0]).run()
    assert _baseline(app) == before


def test_empty_slice_shows_message(app):
    """Сегмента Aviation нет в курортном отеле: предупреждение вместо ключевых цифр, без ошибок.

    Карточки овербукинга остаются: сегмент на них не влияет (решение принимается для всей даты).
    """
    app.sidebar.radio(key="hotel").set_value("курортный").run()
    app.sidebar.multiselect(key="segments").set_value(["Aviation"]).run()
    assert not app.exception
    assert "Без предоплаты" not in [metric.label for metric in app.metric]
    assert app.warning and "нет броней" in app.warning[0].value


def test_extreme_assumptions(app):
    """Крайние значения ползунков допущений: без ошибок, порог модели в подписи пересчитан (0,2 / 0,5)."""
    app.sidebar.slider(key="unsold").set_value(30).run()
    app.sidebar.slider(key="loss").set_value(20).run()
    assert not app.exception
    assert any("не ниже 0,40" in caption.value for caption in app.caption)


def test_empty_months_show_messages(app):
    """Без месяцев: предупреждение вместо цифр и сообщение на вкладке овербукинга, без ошибок."""
    app.sidebar.multiselect(key="months").set_value([]).run()
    assert not app.exception
    assert app.warning and "нет броней" in app.warning[0].value
    assert any("Нет дат заезда" in info.value for info in app.info)


def test_small_slice_readable_money(app):
    """Малый срез (городской, Aviation): деньги в тысячах евро, без «0,00» и «-0»; овербукинг — верхняя оценка."""
    app.sidebar.radio(key="hotel").set_value("городской").run()
    app.sidebar.multiselect(key="segments").set_value(["Aviation"]).run()
    assert not app.exception
    values = {metric.label: metric.value for metric in app.metric}
    assert values["Без предоплаты"].endswith("тыс. €")
    assert not any(value.startswith(("-0", "0,00")) for value in values.values())
    assert "Итог в деньгах (верхняя оценка)" in values
