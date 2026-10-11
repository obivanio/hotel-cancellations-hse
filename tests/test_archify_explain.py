"""Тесты вставки пояснений в схемы archify."""
import json

import pytest

from src.archify_explain import END, EXPLANATIONS_PATH, START, explain_figures, inject, node_ids
from src.plots import FIGURES_DIR

PAGE = '<html><body><svg><g data-node-id="raw"></g><g data-node-id="clean"></g></svg></body></html>'


def test_inject_once_and_idempotent():
    """Блок вставляется перед </body> один раз; повторная вставка заменяет текст, а не дублирует блок."""
    first = inject(PAGE, {"raw": "Исходные данные"})
    second = inject(first, {"raw": "Новый текст </script>"})
    assert first.count(START) == 1 and first.index(END) < first.index("</body>")
    assert second.count(START) == 1 and second.count(END) == 1
    assert "Новый текст <\\/script>" in second and "Исходные данные" not in second


def test_explain_figures_checks_ids(tmp_path):
    """Пояснение к шагу, которого нет в схеме, — ошибка; иначе каждая схема получает один блок."""
    (tmp_path / "s.html").write_text(PAGE, encoding="utf-8")
    texts = tmp_path / "explanations.json"
    texts.write_text(json.dumps({"s": {"raw": "a", "clean": "b"}}), encoding="utf-8")
    assert explain_figures(tmp_path, texts) == {"s.html": 2}
    assert explain_figures(tmp_path, texts) == {"s.html": 2}
    assert (tmp_path / "s.html").read_text(encoding="utf-8").count(START) == 1
    texts.write_text(json.dumps({"s": {"missing": "c"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        explain_figures(tmp_path, texts)


@pytest.mark.parametrize("name, steps", [("19_prepayment_decision", 9), ("20_model_full_pipeline", 16)])
def test_real_figures_have_all_explanations(name, steps):
    """В схемах 19 и 20 есть все шаги из explanations.json, пояснение у каждого шага, блок вставлен один раз."""
    explanations = json.loads(EXPLANATIONS_PATH.read_text(encoding="utf-8"))[name]
    html = (FIGURES_DIR / f"{name}.html").read_text(encoding="utf-8")
    assert len(explanations) == steps
    assert set(explanations) == node_ids(html)
    assert html.count(START) == 1
    assert '<html lang="ru"' in html
