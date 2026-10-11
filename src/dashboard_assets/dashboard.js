"use strict";
// Расчёты дашборда — повторяют эталонные функции src/dashboard.py (тест сверки: tests/test_dashboard_js.py).
// Доля отмен в полосках очереди сглажена внутри групп одинакового риска, как в queue_buckets.
// Суммы считаются накопительно в порядке очереди, как np.cumsum в Python, поэтому цифры совпадают.
const Calc = (() => {
  const BUCKETS = 100;
  const SMALL_MONEY = 100000;

  function cutSize(bookings, pct) {
    return Math.floor((pct * bookings + 50) / 100);
  }

  function modelThreshold(a) {
    return a.loss / (a.unsold + a.loss);
  }

  function proba(data, i) {
    return data.probas[data.p[i]];
  }

  function filterIdx(data, hotel, segments, months) {
    const idx = [];
    for (let i = 0; i < data.h.length; i += 1) {
      if ((hotel === "все отели" || data.hotels[data.h[i]] === hotel)
          && segments.includes(data.segments[data.s[i]]) && months.includes(data.m[i])) idx.push(i);
    }
    return idx;
  }

  function queueOrder(data, idx, column) {
    const key = column === "lead_time" ? (i) => data.lead[i] : (i) => proba(data, i);
    return idx.slice().sort((x, y) => key(y) - key(x));
  }

  function prefix(data, order) {
    const n = order.length;
    const p = { cancelValue: new Float64Array(n + 1), reliableValue: new Float64Array(n + 1),
      cancels: new Float64Array(n + 1), reliable: new Float64Array(n + 1) };
    for (let k = 0; k < n; k += 1) {
      const i = order[k];
      const value = data.value[i];
      const canceled = data.cancel[i] === 1;
      p.cancelValue[k + 1] = p.cancelValue[k] + (canceled ? value : 0);
      p.reliableValue[k + 1] = p.reliableValue[k] + (canceled ? 0 : value);
      p.cancels[k + 1] = p.cancels[k] + (canceled && value > 0 ? 1 : 0);
      p.reliable[k + 1] = p.reliable[k] + (!canceled && value > 0 ? 1 : 0);
    }
    return p;
  }

  function outcome(p, k, a) {
    const last = p.cancels.length - 1;
    const cancels = p.cancels[last];
    const reliable = p.reliable[last];
    const missed = a.unsold * (p.cancelValue[last] - p.cancelValue[k]);
    const falseAlarm = a.loss * p.reliableValue[k];
    const missedCancels = cancels - p.cancels[k];
    return {
      missed, false_alarm: falseAlarm, total: missed + falseAlarm,
      caught: p.cancels[k], touched: p.reliable[k], missed_cancels: missedCancels,
      caught_share: cancels ? p.cancels[k] / cancels : 0,
      touched_share: reliable ? p.reliable[k] / reliable : 0,
      missed_share: cancels ? missedCancels / cancels : 0,
    };
  }

  function queueOutcome(data, idx, pct, a) {
    return outcome(prefix(data, queueOrder(data, idx, "proba")), cutSize(idx.length, pct), a);
  }

  function thresholdPct(data, idx, a) {
    if (!idx.length) return 0;
    const threshold = modelThreshold(a);
    let above = 0;
    for (const i of idx) if (proba(data, i) >= threshold) above += 1;
    return Math.floor((200 * above + idx.length) / (2 * idx.length));
  }

  function keyNumbers(data, idx, a) {
    const p = prefix(data, queueOrder(data, idx, "proba"));
    return {
      bookings: idx.length, threshold: modelThreshold(a), threshold_pct: thresholdPct(data, idx, a),
      baseline: outcome(p, 0, a).total, everyone: outcome(p, idx.length, a).total,
    };
  }

  function queueBuckets(data, order) {
    const n = order.length;
    const groupSize = new Map();
    const groupCanceled = new Map();
    for (const i of order) {
      groupSize.set(data.p[i], (groupSize.get(data.p[i]) || 0) + 1);
      groupCanceled.set(data.p[i], (groupCanceled.get(data.p[i]) || 0) + data.cancel[i]);
    }
    const canceled = new Float64Array(n + 1);
    const risk = new Float64Array(n + 1);
    for (let k = 0; k < n; k += 1) {
      const g = data.p[order[k]];
      canceled[k + 1] = canceled[k] + groupCanceled.get(g) / groupSize.get(g);
      risk[k + 1] = risk[k] + proba(data, order[k]);
    }
    const rows = [];
    for (let i = 0; i < BUCKETS; i += 1) {
      const start = cutSize(n, i);
      const end = cutSize(n, i + 1);
      const size = end - start;
      rows.push({ bookings: size, cancel_share: size ? (canceled[end] - canceled[start]) / size : 0,
        mean_risk: size ? (risk[end] - risk[start]) / size : 0 });
    }
    return rows;
  }

  function shareCurves(data, idx, a) {
    const curves = {};
    for (const [method, column] of [["модель", "proba"], ["срок брони", "lead_time"]]) {
      const p = prefix(data, queueOrder(data, idx, column));
      curves[method] = [];
      for (let pct = 0; pct <= BUCKETS; pct += 1) curves[method].push(outcome(p, cutSize(idx.length, pct), a));
    }
    return curves;
  }

  function heatCells(data, idx) {
    const cells = {};
    for (const i of idx) {
      const key = `${data.segments[data.s[i]]}|${data.m[i]}`;
      const cell = cells[key] || (cells[key] = { bookings: 0, canceled: 0, risk: 0 });
      cell.bookings += 1;
      cell.canceled += data.cancel[i];
      cell.risk += proba(data, i);
    }
    for (const cell of Object.values(cells)) {
      cell.cancel = (cell.canceled / cell.bookings) * 100;
      cell.risk = (cell.risk / cell.bookings) * 100;
    }
    return cells;
  }

  function comma(value, digits = 2) {
    let text = Number(value).toFixed(digits);
    if (Number(text) === 0) text = (0).toFixed(digits);
    return text.replace(".", ",");
  }

  function money(value) {
    if (Math.abs(value) < SMALL_MONEY) return `${comma(value / 1e3, 1)} тыс. €`;
    return `${comma(value / 1e6)} млн €`;
  }

  function count(value) {
    return String(Math.round(value)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  }

  function percent(share) {
    return `${(share * 100).toFixed(0)} %`;
  }

  return { BUCKETS, cutSize, modelThreshold, filterIdx, queueOrder, prefix, outcome, queueOutcome, thresholdPct,
    keyNumbers, queueBuckets, shareCurves, heatCells, comma, money, count, percent };
})();

if (typeof module !== "undefined") module.exports = Calc;

// Отрисовка страницы: работает только в браузере.
if (typeof document !== "undefined") {
  (() => {
    const DATA = JSON.parse(document.getElementById("data").textContent);
    const B = DATA.bookings;
    const $ = (id) => document.getElementById(id);
    const DEFAULTS = DATA.defaults;
    const state = { hotel: "все отели", segments: DATA.segmentOrder.slice(), months: DATA.months.slice(),
      unsold: DEFAULTS.unsold, loss: DEFAULTS.loss, pct: 0, follow: true };
    let slice = null;

    const plural = (n, forms) => {
      const a = Math.abs(n) % 100;
      const b = a % 10;
      if (a > 10 && a < 20) return forms[2];
      if (b === 1) return forms[0];
      return b > 1 && b < 5 ? forms[1] : forms[2];
    };
    const rooms = (n) => `${Calc.count(n)} ${plural(Math.round(n), ["номер", "номера", "номеров"])}`;
    const moneyParts = (value) => {
      const text = Calc.money(value);
      const cut = text.indexOf(" ");
      return [text.slice(0, cut), text.slice(cut + 1)];
    };
    const pct1 = (share) => `${Calc.comma(share * 100, 1)} %`;
    // Неразрывный пробел между числом и единицей: «0,64 млн €» и «90 %» не разрываются при переносе строки.
    const nb = (text) => text.replace(/ (?=млн|тыс|%|€)/g, "\u00A0");
    const unitOf = (max) => (max < 100000 ? { div: 1e3, name: "тыс. €" } : { div: 1e6, name: "млн €" });

    // Подсказка при наведении
    const tip = $("tip");
    const showTip = (html, x, y) => {
      tip.innerHTML = nb(html);
      tip.hidden = false;
      const w = tip.offsetWidth;
      const h = tip.offsetHeight;
      tip.style.left = `${Math.min(Math.max(8, x + 14), window.innerWidth - w - 8)}px`;
      tip.style.top = `${y - h - 14 < 8 ? y + 18 : y - h - 14}px`;
    };
    const hideTip = () => { tip.hidden = true; };

    // Фильтры
    const fillChecks = (box, items, chosen, label) => {
      box.innerHTML = items.map((item) => `<label><input type="checkbox" value="${item}"
        ${chosen.includes(item) ? "checked" : ""}>${label(item)}</label>`).join("");
    };
    fillChecks($("segList"), DATA.segmentOrder, state.segments, (s) => DATA.segmentNames[s]);
    fillChecks($("monList"), DATA.months, state.months, (m) => DATA.monthFull[m]);

    const summaries = () => {
      const segs = state.segments;
      $("segSum").textContent = segs.length === DATA.segmentOrder.length ? "Все сегменты"
        : segs.length === 1 ? DATA.segmentNames[segs[0]]
          : segs.length === 0 ? "Сегмент не выбран"
          : `${segs.length} ${plural(segs.length, ["сегмент", "сегмента", "сегментов"])}`;
      const months = DATA.months.filter((m) => state.months.includes(m));
      $("monSum").textContent = months.length === DATA.months.length ? "Май–август 2017"
        : months.length ? months.map((m) => DATA.monthFull[m]).join(", ") : "Месяц не выбран";
    };

    const readChecks = (box) => [...box.querySelectorAll("input:checked")].map((input) => input.value);
    $("segList").addEventListener("change", () => { state.segments = readChecks($("segList")); refilter(); });
    $("monList").addEventListener("change", () => { state.months = readChecks($("monList")).map(Number); refilter(); });
    document.querySelectorAll("[data-all]").forEach((button) => button.addEventListener("click", () => {
      const box = button.dataset.all === "seg" ? $("segList") : $("monList");
      box.querySelectorAll("input").forEach((input) => { input.checked = true; });
      box.dispatchEvent(new Event("change"));
    }));
    document.querySelectorAll("#hotel button").forEach((button) => button.addEventListener("click", () => {
      document.querySelectorAll("#hotel button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      state.hotel = button.dataset.hotel;
      refilter();
      drawWeeks();
    }));
    const closePicks = (except) => document.querySelectorAll("details.pick[open]").forEach((d) => {
      if (d !== except) d.open = false;
    });
    document.addEventListener("click", (event) => closePicks(event.target.closest("details.pick")));
    document.addEventListener("keydown", (event) => {
      const open = document.querySelector("details.pick[open]");
      if (event.key === "Escape" && open) { open.open = false; open.querySelector("summary").focus(); }
    });

    // Допущения
    const setAssumptions = () => {
      state.unsold = Number($("unsold").value) / 100;
      state.loss = Number($("loss").value) / 100;
      $("unsoldOut").textContent = `${$("unsold").value} %`;
      $("lossOut").textContent = `${$("loss").value} %`;
      update();
    };
    $("unsold").addEventListener("input", setAssumptions);
    $("loss").addEventListener("input", setAssumptions);
    $("reset").addEventListener("click", () => {
      $("unsold").value = Math.round(DEFAULTS.unsold * 100);
      $("loss").value = Math.round(DEFAULTS.loss * 100);
      state.follow = true;
      setAssumptions();
    });

    // Очередь броней
    const queue = $("queue");
    const bars = [];
    for (let n = 0; n < Calc.BUCKETS; n += 1) {
      const bar = document.createElement("i");
      bar.style.setProperty("--h", `${100 - n * 0.45}%`);
      bar.style.setProperty("--n", n);
      queue.appendChild(bar);
      bars.push(bar);
    }
    setTimeout(() => queue.classList.remove("intro"), 1800);
    $("share").addEventListener("input", (event) => {
      state.pct = Number(event.target.value);
      state.follow = false;
      update();
    });
    $("thr").addEventListener("click", () => { state.follow = true; update(); });
    $("queueBox").addEventListener("pointermove", (event) => {
      if (!slice || !slice.idx.length) return;
      const rect = queue.getBoundingClientRect();
      const i = Math.min(Calc.BUCKETS - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * 100)));
      const b = slice.buckets[i];
      showTip(b.bookings ? `<b>${i + 1}-й процент очереди</b><br>${Calc.count(b.bookings)} ${plural(b.bookings,
        ["бронь", "брони", "броней"])}<br>доля отмен: ${Calc.percent(b.cancel_share)}<br>средний риск по модели:
        ${Calc.percent(b.mean_risk)}` : "В этом проценте нет броней", event.clientX, event.clientY);
    });
    $("queueBox").addEventListener("pointerleave", hideTip);

    function refilter() {
      summaries();
      const idx = Calc.filterIdx(B, state.hotel, state.segments, state.months);
      const order = Calc.queueOrder(B, idx, "proba");
      slice = { idx, prefix: Calc.prefix(B, order), buckets: Calc.queueBuckets(B, order) };
      slice.buckets.forEach((b, n) => {
        bars[n].style.setProperty("--c", `${(b.cancel_share * 100).toFixed(2)}%`);
        bars[n].classList.toggle("none", !b.bookings);
      });
      drawHeat();
      update();
    }

    function update() {
      const a = { unsold: state.unsold, loss: state.loss };
      const empty = !slice.idx.length;
      document.querySelectorAll("#outcome, #queueBox, .marks, .ends, .hint").forEach((el) => { el.hidden = empty; });
      $("emptyHero").hidden = !empty;
      setEmpty($("curve"), empty);
      if (empty) return;
      const key = Calc.keyNumbers(B, slice.idx, a);
      if (state.follow) state.pct = key.threshold_pct;
      $("share").value = state.pct;
      $("share").setAttribute("aria-valuetext", `предоплата с ${state.pct} % броней`);
      const res = Calc.outcome(slice.prefix, Calc.cutSize(slice.idx.length, state.pct), a);
      const [value, unit] = moneyParts(res.total);
      $("lossValue").textContent = value;
      $("lossUnit").textContent = unit;
      $("lossNote").textContent = nb(`потерь за май–август 2017 в срезе из ${Calc.count(key.bookings)} ${plural(
        key.bookings, ["брони", "броней", "броней"])}. Без предоплаты — ${Calc.money(key.baseline)}, с предоплатой
        со всех — ${Calc.money(key.everyone)}.`.replace(/\s+/g, " "));
      $("caught").textContent = Calc.count(res.caught);
      $("caughtShare").textContent = nb(Calc.percent(res.caught_share));
      $("touched").textContent = Calc.count(res.touched);
      $("touchedShare").textContent = nb(Calc.percent(res.touched_share));
      $("missed").textContent = Calc.count(res.missed_cancels);
      $("missedShare").textContent = nb(Calc.percent(res.missed_share));
      bars.forEach((bar, n) => bar.classList.toggle("off", n >= state.pct));
      const cut = $("cut");
      cut.style.left = `calc(${state.pct}% - 1px)`;
      cut.classList.toggle("near-start", state.pct < 8);
      cut.classList.toggle("near-end", state.pct > 92);
      $("cutTxt").textContent = `предоплата с ${state.pct} %`;
      const thr = $("thr");
      thr.style.left = `${key.threshold_pct}%`;
      thr.classList.toggle("near-start", key.threshold_pct < 16);
      thr.classList.toggle("near-end", key.threshold_pct > 84);
      thr.textContent = `порог ${Calc.comma(key.threshold)}`;
      thr.setAttribute("aria-label", `Поставить черту на порог ${Calc.comma(key.threshold)}: предоплата с
        ${key.threshold_pct} % броней`.replace(/\s+/g, " "));
      drawCurve(Calc.shareCurves(B, slice.idx, a));
    }

    // Графики
    const setEmpty = (svg, empty) => {
      svg.toggleAttribute("hidden", empty);
      svg.parentElement.querySelector(".chart-empty").hidden = !empty;
    };

    function drawCurve(curves) {
      const model = curves["модель"].map((r) => r.total);
      const lead = curves["срок брони"].map((r) => r.total);
      const unit = unitOf(Math.max(...model, ...lead));
      const top = Math.max(...model, ...lead) / unit.div || 1;
      const step = [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100].find((s) => top / s <= 5) || 100;
      const yMax = Math.ceil(top / step) * step;
      const X = (pct) => 60 + pct * 5.5;
      const Y = (v) => 260 - (v / unit.div / yMax) * 230;
      const path = (k) => k.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
      let s = `<line x1="60" y1="260" x2="610" y2="260" stroke="#DCE2EC"/>`;
      for (let v = 0; v <= yMax + 1e-9; v += step) {
        s += `<text class="t" x="50" y="${Y(v * unit.div) + 4}" text-anchor="end">${Calc.comma(v, step < 1 ? 1 : 0)}</text>`;
      }
      s += `<text class="t" x="60" y="290">0 %</text><text class="t" x="610" y="290" text-anchor="end">100 % броней
        с предоплатой</text><text class="t" x="54" y="12">потери, ${unit.name}</text>`;
      s += `<path d="${path(lead)}" fill="none" stroke="#6B7894" stroke-width="3" stroke-dasharray="8 6"/>`;
      s += `<path d="${path(model)}" fill="none" stroke="#1D3FBF" stroke-width="4"/>`;
      s += `<text class="t strong" x="${X(22)}" y="${Math.min(Y(lead[22]) - 14, 250)}">по сроку брони</text>`;
      s += `<text class="t cob" x="${X(80)}" y="${Y(model[80]) + 26}">по модели</text>`;
      const x = state.pct;
      const ym = Y(model[x]);
      const yl = Y(lead[x]);
      const diff = lead[x] - model[x];
      s += `<line x1="${X(x)}" y1="${yl}" x2="${X(x)}" y2="${ym}" stroke="#E8A317" stroke-width="3"/>`;
      s += `<circle cx="${X(x)}" cy="${ym}" r="7" fill="#1D3FBF"/><circle cx="${X(x)}" cy="${yl}" r="6" fill="#6B7894"/>`;
      const right = x < 55;
      const text = diff >= 0 ? `модель экономит ${Calc.money(diff)}` : `модель дороже на ${Calc.money(-diff)}`;
      s += `<text class="t saf" x="${X(x) + (right ? 12 : -12)}" y="${Math.min(Math.max(yl, ym) + 28, 250)}"
        text-anchor="${right ? "start" : "end"}">${text}</text>`;
      $("curve").innerHTML = s;
    }

    function drawWeeks() {
      const w = DATA.weeks[state.hotel];
      const t = DATA.obTotals[state.hotel];
      const max = Math.max(...w.actual, ...w.cautious, ...w.bench) || 1;
      const H = (v) => (v / max) * 176;
      const gap = 560 / w.actual.length;
      let s = `<line x1="40" y1="260" x2="620" y2="260" stroke="#DCE2EC"/>`;
      w.actual.forEach((f, i) => {
        const x = 50 + i * gap;
        s += `<rect data-i="${i}" x="${x}" y="${260 - H(f)}" width="${gap * 0.66}" height="${H(f)}" rx="6" fill="#C9D3EA"/>`;
        if (i === 0 || w.month[i] !== w.month[i - 1]) {
          s += `<text class="t" x="${x}" y="282">${DATA.monthShort[w.month[i]]}</text>`;
        }
      });
      const line = (k) => k.map((v, i) => `${(50 + i * gap + gap * 0.33).toFixed(1)},${(260 - H(v)).toFixed(1)}`).join(" ");
      s += `<polyline points="${line(w.bench)}" fill="none" stroke="#6B7894" stroke-width="2.5" stroke-dasharray="7 5"/>`;
      s += `<polyline points="${line(w.cautious)}" fill="none" stroke="#1D3FBF" stroke-width="4" stroke-linejoin="round"/>`;
      s += `<text class="t strong" x="40" y="16">столбик — сколько броней отменили на самом деле</text>`;
      s += `<text class="t cob" x="40" y="40">линия — сколько номеров продаём сверх брони</text>`;
      s += `<text class="t" x="40" y="64">пунктир — та же оценка без модели</text>`;
      $("weeks").innerHTML = s;
      const both = state.hotel === "все отели" ? "два отеля, каждый считает сам; " : "";
      const split = state.hotel === "все отели"
        ? `<p>Городской — ${rooms(DATA.obTotals["City Hotel"].per_day)}, курортный —
          ${rooms(DATA.obTotals["Resort Hotel"].per_day)}. По датам — от ${Calc.count(t.per_day_min)}
          до ${Calc.count(t.per_day_max)}, смотря сколько на дату броней и насколько они рискованны.</p>`
        : `<p>Это в среднем на дату заезда. По датам — от ${Calc.count(t.per_day_min)}
          до ${Calc.count(t.per_day_max)}, смотря сколько на дату броней и насколько они рискованны.</p>`;
      $("obAnswer").innerHTML = nb(`<div class="n">≈&nbsp;${Calc.count(t.per_day)}
        <small>${plural(Math.round(t.per_day), ["номер", "номера", "номеров"])} в день сверх брони</small></div>
        ${split}`);
      $("obTotals").innerHTML = nb(`<li><b>${Calc.count(t.resold)}</b>номеров перепродано</li>
        <li><b>${Calc.count(t.walked)}</b>${plural(t.walked, ["гостю", "гостям", "гостям"])} не хватило номера</li>
        <li><b>${pct1(t.covered)}</b>дат, когда номера хватило всем (${both}без модели —
          ${pct1(t.covered_bench)})</li>`);
    }
    $("weeks").addEventListener("pointermove", (event) => {
      const i = event.target.dataset && event.target.dataset.i;
      if (i === undefined) { hideTip(); return; }
      const w = DATA.weeks[state.hotel];
      showTip(`<b>${w.label[i]}</b><br>отменили: ${Calc.count(w.actual[i])}<br>продаём сверх брони:
        ${Calc.count(w.cautious[i])}<br>без модели: ${Calc.count(w.bench[i])}`, event.clientX, event.clientY);
    });
    $("weeks").addEventListener("pointerleave", hideTip);

    const heatColor = (v) => {
      const t = Math.min(v / 60, 1);
      const mix = (from, to) => Math.round(from + (to - from) * t);
      return `rgb(${mix(255, 232)},${mix(246, 163)},${mix(224, 23)})`;
    };
    function drawHeat() {
      const svg = $("heat");
      const segs = DATA.segmentOrder.filter((seg) => state.segments.includes(seg));
      const months = DATA.months.filter((m) => state.months.includes(m));
      const empty = !slice || !slice.idx.length || !segs.length || !months.length;
      setEmpty(svg, empty);
      if (empty) return;
      const cells = Calc.heatCells(B, slice.idx);
      const colW = Math.min(110, 456 / months.length);
      svg.setAttribute("viewBox", `0 0 640 ${segs.length * 46 + 30}`);
      let h = "";
      months.forEach((m, c) => {
        h += `<text class="t" x="${184 + c * colW + (colW - 6) / 2}" y="14" text-anchor="middle">${DATA.monthFull[m]}</text>`;
      });
      segs.forEach((seg, r) => {
        h += `<text class="t strong row" x="0" y="${r * 46 + 52}">${DATA.segmentNames[seg]}</text>`;
        months.forEach((m, c) => {
          const cell = cells[`${seg}|${m}`];
          const x = 184 + c * colW;
          const y = r * 46 + 26;
          if (!cell) {
            h += `<rect x="${x}" y="${y}" width="${colW - 6}" height="40" rx="10" fill="#F3F5F7"/>`;
            h += `<text class="t" x="${x + (colW - 6) / 2}" y="${y + 25}" text-anchor="middle">—</text>`;
            return;
          }
          h += `<rect data-seg="${seg}" data-m="${m}" x="${x}" y="${y}" width="${colW - 6}" height="40" rx="10"
            fill="${heatColor(cell.cancel)}"/>`;
          h += `<text class="t strong" x="${x + (colW - 6) / 2}" y="${y + 25}" text-anchor="middle"
            pointer-events="none">${Calc.comma(cell.cancel, 0)} %</text>`;
        });
      });
      svg.innerHTML = h;
    }
    $("heat").addEventListener("pointermove", (event) => {
      const { seg, m } = event.target.dataset || {};
      if (!seg) { hideTip(); return; }
      const cell = Calc.heatCells(B, slice.idx)[`${seg}|${m}`];
      showTip(`<b>${DATA.segmentNames[seg]}, ${DATA.monthFull[m]}</b><br>${Calc.count(cell.bookings)} ${plural(
        cell.bookings, ["бронь", "брони", "броней"])}<br>отменили: ${Calc.percent(cell.cancel / 100)}<br>средний
        риск по модели: ${Calc.percent(cell.risk / 100)}`, event.clientX, event.clientY);
    });
    $("heat").addEventListener("pointerleave", hideTip);

    // Вкладки схем: мышь и стрелки
    const tabs = [...document.querySelectorAll("#schemeTabs [role=tab]")];
    const selectTab = (tab) => {
      tabs.forEach((t) => {
        t.setAttribute("aria-selected", String(t === tab));
        t.tabIndex = t === tab ? 0 : -1;
      });
      const frame = $("scheme");
      frame.src = `${tab.dataset.src}?theme=light`;
      frame.style.setProperty("--k", tab.dataset.k);
      frame.style.setProperty("--pad", `${tab.dataset.pad}px`);
      frame.title = `Схема: ${tab.textContent.toLowerCase()}`;
      frame.setAttribute("aria-labelledby", tab.id);
    };
    tabs.forEach((tab, i) => {
      tab.addEventListener("click", () => selectTab(tab));
      tab.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        const next = tabs[(i + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
        next.focus();
        selectTab(next);
      });
    });

    // Что делать бизнесу: цифры шага 4 при допущениях по умолчанию
    const v = DATA.verdict;
    const mln = (x) => Calc.comma(x / 1e6);
    $("recs").innerHTML = nb(`
      <div class="rec"><div><div class="n">${mln(v.model)} <small>млн €</small></div>
        <div class="cap">потерь вместо ${mln(v.baseline)}</div></div>
        <div><h3>Брать предоплату с броней, где риск отмены выше ${Calc.comma(Calc.modelThreshold(DEFAULTS))}</h3>
        <p>Это ${Calc.percent(v.model_prepay)} броней, среди них ${Calc.percent(v.model_caught)} будущих отмен.
          Дешевле, чем предоплата со всех (${mln(v.everyone)} млн €).</p>
        <p>Цена решения: предоплату внесут ${Calc.percent(v.model_touched)} гостей, которые приехали бы и так.</p></div></div>
      <div class="rec"><div><div class="n">${mln(v.top30)} <small>млн €</small></div>
        <div class="cap">потерь вместо ${mln(v.baseline)}</div></div>
        <div><h3>Если важнее не раздражать гостей — только с 30 % самых рискованных броней</h3>
        <p>Задеваем ${Calc.percent(v.top30_touched)} надёжных гостей, но ловим только ${Calc.percent(v.top30_caught)}
          отмен.</p></div></div>
      <div class="rec"><div><div class="n">${Calc.comma(v.covered_all * 100, 1)} <small>%</small></div>
        <div class="cap">дат, когда номера хватило всем, — в среднем по двум отелям</div></div>
        <div><h3>Продавать сверх брони по осторожной оценке, в городском отеле — меньше</h3>
        <p>Каждый отель решает сам: на каждую дату продаёт столько лишних номеров, сколько отмен случится
          с вероятностью 90 %. В среднем это ${rooms(DATA.obTotals["все отели"].per_day)} в день
          на два отеля: ${Calc.count(DATA.obTotals["City Hotel"].per_day)} в городском
          и ${Calc.count(DATA.obTotals["Resort Hotel"].per_day)} в курортном.</p>
        <p>В курортном отеле номера хватило в ${pct1(v.covered_resort)} дат, в городском — только
          в ${pct1(v.covered_city)}: там отменяют группами, разом. В городском отеле стоит продавать меньше
          лишних номеров, чем советует оценка.</p></div></div>`);

    refilter();
    drawWeeks();
  })();
}
