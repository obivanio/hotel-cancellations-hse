"""Пояснения простыми словами в паспорте шага схем archify (reports/figures/19_…html, 20_…html).

Скрипт в HTML схемы следит за открытым паспортом (адрес #focus=<id шага>) и дописывает под названием шага абзац
из diagrams/explanations.json. Стиль блока ещё держит метку направления связи («ВЫХ →») в одну строку, чтобы она
не наезжала на название соседнего шага. Повторная вставка заменяет прежнюю, а не дублирует её.
Запуск: python -m src.archify_explain
"""
import json
import re

from src.pipeline import ROOT
from src.plots import FIGURES_DIR

EXPLANATIONS_PATH = ROOT / "diagrams" / "explanations.json"
START, END = "<!-- hse-explain:start -->", "<!-- hse-explain:end -->"
SCRIPT = """<style>.hse-explain{margin:.5rem 0 .45rem;font-size:.75rem;line-height:1.5;white-space:normal;
color:var(--text)}
.relationship-lens-row{grid-template-columns:2.7rem minmax(0,1fr)}
.relationship-lens-direction{white-space:nowrap}</style>
<script>(function(){var EX=__EXPLANATIONS__;
function apply(){var chip=document.getElementById('focus-chip');if(!chip)return;
var m=location.hash.match(/focus=([^&]+)/);var text=m&&EX[decodeURIComponent(m[1])];
var p=chip.querySelector('.hse-explain');if(!text){if(p)p.remove();return;}
if(!p){var title=chip.querySelector('.semantic-passport-detail')||chip.querySelector('.relationship-lens-title');
if(!title)return;p=document.createElement('p');p.className='hse-explain';title.after(p);}
if(p.textContent!==text)p.textContent=text;}
new MutationObserver(apply).observe(document.documentElement,{subtree:true,childList:true});
window.addEventListener('hashchange',apply);})();</script>"""


def node_ids(html: str) -> set[str]:
    """Id шагов схемы archify (атрибут data-node-id)."""
    return set(re.findall(r'data-node-id="([A-Za-z0-9_-]+)"', html))


def inject(html: str, explanations: dict[str, str]) -> str:
    """HTML схемы с блоком пояснений перед </body>; прежний блок (между метками START и END) заменяется."""
    payload = json.dumps(explanations, ensure_ascii=False).replace("</", "<\\/")
    block = f"{START}\n{SCRIPT.replace('__EXPLANATIONS__', payload)}\n{END}\n"
    html = re.sub(f"{re.escape(START)}.*?{re.escape(END)}\n?", "", html, flags=re.S)
    return html.replace("</body>", block + "</body>", 1)


def explain_figures(figures_dir=FIGURES_DIR, path=EXPLANATIONS_PATH) -> dict[str, int]:
    """Вставить пояснения во все схемы из explanations.json; вернуть число пояснённых шагов по файлам.

    Пояснение к шагу, которого нет в схеме, — ошибка: значит, схему пересобрали, а тексты не обновили.
    """
    counts = {}
    for name, explanations in json.loads(path.read_text(encoding="utf-8")).items():
        figure = figures_dir / f"{name}.html"
        html = figure.read_text(encoding="utf-8")
        missing = set(explanations) - node_ids(html)
        if missing:
            raise ValueError(f"{figure.name}: нет шагов {sorted(missing)}")
        figure.write_text(inject(html, explanations), encoding="utf-8")
        counts[figure.name] = len(explanations)
    return counts


if __name__ == "__main__":
    for figure_name, steps in explain_figures().items():
        print(f"{figure_name}: пояснений {steps}")
