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

  function segmentTable(data, idx, a) {
    const groups = new Map();
    idx.forEach((i) => {
      const code = data.segments[data.s[i]];
      if (!groups.has(code)) groups.set(code, []);
      groups.get(code).push(i);
    });
    const rows = [...groups.entries()].map(([segment, part]) => {
      const key = keyNumbers(data, part, a);
      const model = queueOutcome(data, part, key.threshold_pct, a).total;
      const canceled = part.reduce((sum, i) => sum + data.cancel[i], 0);
      return { segment, bookings: part.length, cancel_share: canceled / part.length, baseline: key.baseline, model,
        saving: key.baseline - model };
    });
    return rows.sort((x, y) => (y.saving - x.saving) || (x.segment < y.segment ? -1 : x.segment > y.segment ? 1 : 0));
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
    keyNumbers, segmentTable, queueBuckets, shareCurves, heatCells, comma, money, count, percent };
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
    // В показателях и таблице число не разрывается и внутри: «4 898» держится одной строкой.
    const tight = (text) => nb(text).replace(/(\d) (?=\d)/g, "$1\u00A0");
    const unitOf = (max) => (max < 100000 ? { div: 1e3, name: "тыс. €" } : { div: 1e6, name: "млн €" });

    const clone = (value) => JSON.parse(JSON.stringify(value));

    // Панель Plotly по-русски: свой словарь, без загрузки локалей из сети.
    const RU = {
      "Download plot as a PNG": "Скачать PNG", "Download plot": "Скачать", "Zoom": "Приблизить", "Pan": "Сдвинуть",
      "Zoom in": "Крупнее", "Zoom out": "Мельче", "Reset axes": "Вернуть масштаб", "Reset": "Вернуть масштаб",
      "Autoscale": "Вписать", "Double-click to zoom back out": "Двойной щелчок — вернуть масштаб",
      "Double-click on legend to isolate one trace": "Двойной щелчок по легенде — оставить одну линию",
      "Preparing image - this may take a few seconds": "Готовим картинку…",
      "Image download succeeded": "Картинка сохранена",
      "Sorry, there was a problem downloading your image!": "Не удалось сохранить картинку",
    };
    Plotly.register({ moduleType: "locale", name: "ru", format: { decimal: ",", thousands: " " }, dictionary: RU });

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
      redrawAll();
    };
    $("unsold").addEventListener("input", setAssumptions);
    $("loss").addEventListener("input", setAssumptions);
    $("reset").addEventListener("click", () => {
      $("unsold").value = Math.round(DEFAULTS.unsold * 100);
      $("loss").value = Math.round(DEFAULTS.loss * 100);
      state.follow = true;
      setAssumptions();
    });

    // Очередь броней: черту двигает ползунок под графиком (мышью или стрелками); ползунки двух видов — одна черта
    ["share", "dShare"].forEach((id) => $(id).addEventListener("input", (event) => {
      state.pct = Number(event.target.value);
      state.follow = false;
      redrawAll();
    }));
    $("thr").addEventListener("click", () => { state.follow = true; redrawAll(); });

    function refilter() {
      summaries();
      const idx = Calc.filterIdx(B, state.hotel, state.segments, state.months);
      const order = Calc.queueOrder(B, idx, "proba");
      slice = { idx, prefix: Calc.prefix(B, order), buckets: Calc.queueBuckets(B, order) };
      redrawAll();
    }

    // Состояние общее для двух видов; рисуется только видимый вид (Plotly в скрытом блоке получает нулевую ширину),
    // скрытый перерисуется при показе.
    function redrawAll() {
      const a = { unsold: state.unsold, loss: state.loss };
      const empty = !slice.idx.length;
      const key = empty ? null : Calc.keyNumbers(B, slice.idx, a);
      if (key && state.follow) state.pct = key.threshold_pct;
      const res = key && Calc.outcome(slice.prefix, Calc.cutSize(slice.idx.length, state.pct), a);
      const curves = key && Calc.shareCurves(B, slice.idx, a);
      if (views.dashboard.hidden) drawPresentation(key, res, curves);
      else drawDashboard(key, res, curves, a);
    }

    function drawPresentation(key, res, curves) {
      const empty = !key;
      drawHeat("pHeat");
      drawWeeks("pWeeks");
      drawObText();
      document.querySelectorAll("#outcome, #queueBox, .hint").forEach((el) => { el.hidden = empty; });
      $("emptyHero").hidden = !empty;
      setEmpty($("pCurves"), empty);
      if (empty) return;
      $("share").value = state.pct;
      $("share").setAttribute("aria-valuetext", `предоплата с ${state.pct} % броней`);
      $("share").style.setProperty("--p", `${state.pct}%`);
      $("shareTxt").textContent = nb(`${state.pct} %`);
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
      const thr = $("thr");
      thr.style.left = `${key.threshold_pct}%`;
      thr.classList.toggle("near-start", key.threshold_pct < 16);
      thr.classList.toggle("near-end", key.threshold_pct > 84);
      thr.textContent = `порог ${Calc.comma(key.threshold)}`;
      thr.setAttribute("aria-label", `Поставить черту на порог ${Calc.comma(key.threshold)}: предоплата с
        ${key.threshold_pct} % броней`.replace(/\s+/g, " "));
      drawQueue("pQueue");
      drawCurves("pCurves", curves);
    }

    // Вид «Дашборд»: первые пять показателей — от фильтров, допущений и черты; последние три — только от отеля.
    function drawDashboard(key, res, curves, a) {
      const t = DATA.obTotals[state.hotel];
      const set = (id, text) => { $(id).textContent = tight(text); };
      set("kRooms", `≈ ${Calc.count(t.per_day)}`);
      set("kRoomsNote", `по датам ${Calc.count(t.per_day_min)}–${Calc.count(t.per_day_max)}`);
      set("kWalked", Calc.count(t.walked));
      set("kWalkedNote", `перепродано ${Calc.count(t.resold)}`);
      set("kCovered", Calc.comma(t.covered * 100, 1));
      set("kCoveredNote", `без модели ${pct1(t.covered_bench)}`);
      $("dWeeksSub").textContent = state.hotel === "все отели" ? "Оба отеля, каждый считает сам."
        : `${state.hotel === "City Hotel" ? "Городской" : "Курортный"} отель.`;
      drawWeeks("dWeeks");
      drawHeat("dHeat");
      const empty = !key;
      setEmpty($("dQueue"), empty);
      $("dTrack").hidden = empty;
      setEmpty($("dCurves"), empty);
      setEmpty($("dSegments"), empty);
      if (empty) {
        ["kLoss", "kPct", "kCaught", "kTouched", "kSave"].forEach((id) => set(id, "—"));
        ["kLossUnit", "kLossNote", "kPctUnit", "kPctNote", "kCaughtNote", "kTouchedNote", "kSaveUnit",
          "dSegmentsSub"].forEach((id) => set(id, ""));
        $("dTable").innerHTML = `<tr><td class="none" colspan="6">В этом срезе нет броней. Выберите другой сегмент
          или месяц.</td></tr>`;
        return;
      }
      const [loss, lossUnit] = moneyParts(res.total);
      set("kLoss", loss);
      set("kLossUnit", lossUnit);
      set("kLossNote", `без предоплаты ${Calc.money(key.baseline)}\nсо всех ${Calc.money(key.everyone)}`);
      set("kPct", `${state.pct}`);
      set("kPctUnit", "% броней");
      set("kPctNote", `порог риска ${Calc.comma(key.threshold)}`);
      set("kCaught", Calc.percent(res.caught_share));
      set("kCaughtNote", `${Calc.count(res.caught)} ${plural(res.caught, ["бронь", "брони", "броней"])}`);
      set("kTouched", Calc.percent(res.touched_share));
      set("kTouchedNote", `${Calc.count(res.touched)} ${plural(res.touched, ["бронь", "брони", "броней"])}`);
      const [save, saveUnit] = moneyParts(curves["срок брони"][state.pct].total - curves["модель"][state.pct].total);
      set("kSave", save);
      set("kSaveUnit", saveUnit);
      $("dShare").value = state.pct;
      $("dShare").setAttribute("aria-valuetext", `предоплата с ${state.pct} % броней`);
      $("dShare").style.setProperty("--p", `${state.pct}%`);
      $("dShareTxt").textContent = nb(`${state.pct} % броней`);
      drawQueue("dQueue");
      drawCurves("dCurves", curves);
      const rows = Calc.segmentTable(B, slice.idx, a);
      const unit = unitOf(Math.max(...rows.map((r) => r.baseline), 1));
      $("dSegmentsSub").textContent = `Без предоплаты и с предоплатой по порогу риска ${Calc.comma(key.threshold)}.`;
      plot("dSegments", "segments", (data, layout) => {
        const names = rows.map((r) => DATA.segmentNames[r.segment]);
        data[0].y = names;
        data[1].y = names;
        data[0].x = rows.map((r) => r.baseline / unit.div);
        data[1].x = rows.map((r) => r.model / unit.div);
        data.forEach((trace) => { trace.hovertemplate = `%{y}: %{x:.2f} ${unit.name}<extra>${trace.name}</extra>`; });
        layout.xaxis.title.text = `потери, ${unit.name}`;
        // Высота — по числу сегментов в срезе, как у тепловой карты: один сегмент не растягивается на весь график;
        // легенда — над областью графика, чтобы не лечь на столбики низкого графика.
        Object.assign(layout.legend, { y: 1.02, yanchor: "bottom" });
        layout.margin.t = Math.max(layout.margin.t, 36);
        const frame = layout.margin.t + layout.margin.b;
        layout.height = frame + Math.round(((layout.height - frame) * Math.max(rows.length, 2)) / DATA.segmentOrder.length);
      });
      // Одна единица на всю таблицу (в заголовках): числа в столбце сравниваются напрямую.
      const money = (value) => Calc.comma(value / unit.div, unit.div === 1e6 ? 2 : 1);
      document.querySelectorAll("#viewDashboard .unit").forEach((el) => { el.textContent = nb(`, ${unit.name}`); });
      $("dTable").innerHTML = rows.map((r) => tight(`<tr><td>${DATA.segmentNames[r.segment]}</td>
        <td>${Calc.count(r.bookings)}</td><td>${Calc.percent(r.cancel_share)}</td><td>${money(r.baseline)}</td>
        <td>${money(r.model)}</td><td class="save">${money(r.saving)}</td></tr>`)).join("");
    }

    // Графики Plotly: вид задаёт Python (DATA.figures), здесь меняются только данные и положение черты/точки.
    const setEmpty = (el, empty) => {
      el.toggleAttribute("hidden", empty);
      el.parentElement.querySelector(".chart-empty").hidden = !empty;
    };
    // narrow — узкий экран (телефон): длинная подпись оси и частые подписи недель там не помещаются.
    function plot(id, name, patch) {
      const el = $(id);
      if (!el) return;
      const data = clone(DATA.figures[name].data);
      const layout = clone(DATA.figures[name].layout);
      const narrow = el.clientWidth < 560;
      patch(data, layout, narrow);
      // На узком экране легенда переносится на 2–3 строки: ставим её над областью графика и оставляем ей место сверху.
      if (narrow && layout.showlegend) {
        Object.assign(layout.legend, { y: 1.02, yanchor: "bottom" });
        layout.margin.t = Math.max(layout.margin.t, 20 * data.filter((t) => t.showlegend !== false).length);
      }
      if (el.dataset.height) layout.height = Number(el.dataset.height);
      Plotly.react(el, data, layout, DATA.plotlyConfig);
    }

    // Полоска n (с нуля) левее черты, если n < pct; черта стоит между столбиками pct и pct + 1 (ось x — 1..100).
    // Процент очереди без броней (малый срез) остаётся пустым местом.
    function drawQueue(id) {
      plot(id, "queue", (data, layout, narrow) => {
        const share = slice.buckets.map((b) => (b.bookings ? b.cancel_share * 100 : null));
        const opacity = share.map((_, n) => (n < state.pct ? 1 : 0.45));
        data[0].y = share;
        data[0].customdata = slice.buckets.map((b) => [b.bookings, b.mean_risk * 100]);
        data[0].marker.opacity = opacity;
        data[1].y = share.map((v) => (v === null ? null : 100 - v));
        data[1].marker.opacity = opacity;
        layout.shapes[0].x0 = state.pct + 0.5;
        layout.shapes[0].x1 = state.pct + 0.5;
        layout.annotations[0].x = state.pct + 0.5;
        layout.annotations[0].text = `предоплата с ${state.pct} %`;
        layout.annotations[0].xanchor = state.pct < 8 ? "left" : state.pct > 92 ? "right" : "center";
        if (narrow) { layout.margin.b = 16; layout.xaxis.title.text = ""; }
      });
    }

    function drawCurves(id, curves) {
      plot(id, "curves", (data, layout) => {
        const model = curves["модель"].map((r) => r.total);
        const lead = curves["срок брони"].map((r) => r.total);
        const unit = unitOf(Math.max(...model, ...lead));
        data[0].y = lead.map((v) => v / unit.div);
        data[1].y = model.map((v) => v / unit.div);
        data[2].x = [state.pct];
        data[2].y = [model[state.pct] / unit.div];
        [0, 1].forEach((k) => { data[k].hovertemplate = data[k].hovertemplate.replace("млн €", unit.name); });
        layout.yaxis.title.text = `потери, ${unit.name}`;
        const diff = lead[state.pct] - model[state.pct];
        Object.assign(layout.annotations[0], { x: state.pct, y: model[state.pct] / unit.div,
          text: diff >= 0 ? `модель экономит ${Calc.money(diff)}` : `модель дороже на ${Calc.money(-diff)}`,
          xanchor: state.pct < 55 ? "left" : "right" });
      });
    }

    function drawWeeks(id) {
      const w = DATA.weeks[state.hotel];
      plot(id, "weeks", (data, layout, narrow) => {
        data.forEach((trace) => { trace.x = w.label; });
        data[0].y = w.actual;
        data[1].y = w.cautious;
        data[2].y = w.bench;
        if (narrow) layout.xaxis.nticks = 3;
      });
    }

    // Ответ «сколько номеров продать сверх брони» — от выбора отеля (R1: «оба» — сумма решений двух отелей).
    function drawObText() {
      const t = DATA.obTotals[state.hotel];
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

    // Высота тепловой карты — по числу выбранных сегментов (у полного списка — как задано в Python);
    // число броней и средний риск клетки — в подсказке.
    function drawHeat(id) {
      const segs = DATA.segmentOrder.filter((seg) => state.segments.includes(seg));
      const months = DATA.months.filter((m) => state.months.includes(m));
      const empty = !slice || !slice.idx.length || !segs.length || !months.length;
      setEmpty($(id), empty);
      if (empty) return;
      const cells = Calc.heatCells(B, slice.idx);
      const cell = (seg, m) => cells[`${seg}|${m}`];
      plot(id, "heatmap", (data, layout) => {
        data[0].y = segs.map((seg) => DATA.segmentNames[seg]);
        data[0].x = months.map((m) => DATA.monthShort[m]);
        data[0].z = segs.map((seg) => months.map((m) => (cell(seg, m) ? cell(seg, m).cancel : null)));
        data[0].text = data[0].z.map((row) => row.map((v) => (v === null ? "–" : `${Calc.comma(v, 0)} %`)));
        data[0].customdata = segs.map((seg) => months.map((m) => (cell(seg, m)
          ? [Calc.count(cell(seg, m).bookings), cell(seg, m).risk] : [0, null])));
        const frame = layout.margin.t + layout.margin.b;
        layout.height = frame + Math.round(((layout.height - frame) * Math.max(segs.length, 2)) / DATA.segmentOrder.length);
      });
    }


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

    // Переключатель «Презентация / Дашборд»: мышь, стрелки, адрес #dashboard
    const views = { presentation: $("viewPresentation"), dashboard: $("viewDashboard") };
    const viewTabs = [...document.querySelectorAll("#viewTabs [role=tab]")];
    const setView = (name) => {
      Object.entries(views).forEach(([view, el]) => { el.hidden = view !== name; });
      viewTabs.forEach((tab) => {
        const on = tab.dataset.view === name;
        tab.setAttribute("aria-selected", String(on));
        tab.tabIndex = on ? 0 : -1;
      });
    };
    function showView(name, push) {
      setView(name);
      if (push) history.replaceState(null, "", name === "dashboard" ? "#dashboard" : location.pathname + location.search);
      redrawAll();
      views[name].querySelectorAll(".plot").forEach((el) => { if (el.data) Plotly.Plots.resize(el); });
    }
    viewTabs.forEach((tab, i) => {
      tab.addEventListener("click", () => showView(tab.dataset.view, true));
      tab.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        const next = viewTabs[(i + (event.key === "ArrowRight" ? 1 : viewTabs.length - 1)) % viewTabs.length];
        next.focus();
        showView(next.dataset.view, true);
      });
    });
    window.addEventListener("hashchange", () => showView(location.hash === "#dashboard" ? "dashboard" : "presentation"));

    // Подстройки под узкий экран считаются при отрисовке, поэтому после изменения размера окна вид перерисовывается.
    let resizeTimer = 0;
    const redrawSoon = () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(redrawAll, 200); };
    window.addEventListener("resize", redrawSoon);
    window.addEventListener("orientationchange", redrawSoon);

    setView(location.hash === "#dashboard" ? "dashboard" : "presentation");
    refilter();
  })();
}
