"""Контраст текста дашборда по WCAG 2.1: обычный текст ≥ 4,5, крупный (от 24 px или от 18,7 px жирным) ≥ 3."""
import re

import pytest

from src.html_dashboard import ASSETS_DIR

TOKENS = dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9A-Fa-f]{6})", (ASSETS_DIR / "dashboard.css").read_text("utf-8")))
TOKENS |= {"white": "#FFFFFF", "heat-low": "#FFF6E0", "heat-high": "#E8A317"}
PAIRS = [
    ("ink", "limestone", False), ("ink", "tile", False), ("ink-2", "limestone", False), ("ink-2", "tile", False),
    ("cobalt", "tile", False), ("cobalt", "limestone", True), ("saffron-ink", "tile", False),
    ("saffron-ink", "limestone", True), ("white", "ink", False), ("night-text", "ink", False),
    ("night-muted", "ink", False), ("saffron", "ink", True), ("ink", "heat-low", False), ("ink", "heat-high", False),
]


def luminance(color: str) -> float:
    """Относительная яркость цвета #RRGGBB по WCAG."""
    channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(text: str, background: str) -> float:
    """Контраст двух цветов: (светлее + 0,05) / (темнее + 0,05)."""
    light, dark = sorted([luminance(text), luminance(background)], reverse=True)
    return (light + 0.05) / (dark + 0.05)


def test_contrast_formula():
    """Чёрный на белом — 21, белый на белом — 1."""
    assert contrast("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast("#FFFFFF", "#FFFFFF") == pytest.approx(1.0)


@pytest.mark.parametrize("text, background, large", PAIRS)
def test_text_contrast(text, background, large):
    """Каждая пара «текст / фон» дашборда проходит порог WCAG AA."""
    assert contrast(TOKENS[text], TOKENS[background]) >= (3.0 if large else 4.5)
