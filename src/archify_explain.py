"""Пояснения простыми словами в паспорте шага схем archify (reports/figures/19_…html, 20_…html).

Скрипт в HTML схемы следит за открытым паспортом шага и дописывает под названием шага абзац
из diagrams/explanations.json на всю ширину паспорта. Id шага берётся из самого паспорта (поле #focus-id), а не из
адреса: при выбранной связи (#relation=…) паспорт шага открыт, и пояснение остаётся. Следим только за паспортом:
archify не пересоздаёт его, а лишь меняет содержимое. Внутри дашборда (схема открыта во фрейме) список шагов
под схемой скрыт: он повторяет схему и не помещается в окно. Повторная вставка заменяет прежнюю, а не дублирует её.
Запуск: python -m src.archify_explain
"""
import json
import re

from src.pipeline import ROOT
from src.plots import FIGURES_DIR

EXPLANATIONS_PATH = ROOT / "diagrams" / "explanations.json"
START, END = "<!-- hse-explain:start -->", "<!-- hse-explain:end -->"
SCRIPT = """<style>.relationship-lens-head{row-gap:0;grid-template-rows:auto auto 1fr}
.relationship-lens-copy{display:contents}.relationship-lens-copy>*{grid-column:1/-1}
.relationship-lens-copy>.relationship-lens-eyebrow,.relationship-lens-copy>.relationship-lens-title,
.relationship-lens-copy>.semantic-passport-detail{grid-column:1}.relationship-lens-actions{grid-column:2;grid-row:1/4}
.hse-explain{margin:.5rem 0 0;font-size:.75rem;line-height:1.5;white-space:normal;color:var(--text)}
html.hse-in-frame .reader-rail{display:none}</style>
<script>(function(){if(window.self!==window.top)document.documentElement.classList.add('hse-in-frame');
var EX=__EXPLANATIONS__;var chip=document.getElementById('focus-chip');
function stepId(){var code=chip.querySelector('#focus-id');if(code)return code.textContent.trim();
var m=location.hash.match(/focus=([^&]+)/);return m?decodeURIComponent(m[1]):'';}
function apply(){var id=stepId();var text=Object.prototype.hasOwnProperty.call(EX,id)&&EX[id];
var p=chip.querySelector('.hse-explain');if(!text){if(p)p.remove();return;}
if(!p){var title=chip.querySelector('.semantic-passport-detail')||chip.querySelector('.relationship-lens-title');
if(!title)return;p=document.createElement('p');p.className='hse-explain';title.after(p);}
if(p.textContent!==text)p.textContent=text;}
if(!chip)return;apply();
new MutationObserver(apply).observe(chip,{subtree:true,childList:true,characterData:true});
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
