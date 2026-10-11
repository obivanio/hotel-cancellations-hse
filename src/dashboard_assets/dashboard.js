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
