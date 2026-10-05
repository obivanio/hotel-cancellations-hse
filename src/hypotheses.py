"""Гипотезы H1–H6: проверка, сводная таблица с поправкой Холма и перепроверка устойчивости."""
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.cleaning import VARIANTS
from src.eda import lead_time_bins
from src.pipeline import ROOT, build_dataset
from src.stats import ALPHA, StatResult, chi2_independence, holm, kruskal_wallis, spearman, two_proportion_ztest

HYPOTHESES_PATH = ROOT / "reports" / "hypotheses.csv"
SMALL_SEGMENTS = ("Aviation", "Complementary")
OTHER_SEGMENT = "Прочие"
NON_REFUND = "Non Refund"
STATEMENTS = {
    "H1": "Чем раньше бронируют, тем чаще отменяют",
    "H2": "Сегменты продаж различаются по доле отмен",
    "H3": "Гости со спецзапросами отменяют реже",
    "H4": "Повторные гости отменяют реже",
    "H5": "Городской отель теряет на отменах больше курортного",
    "H6": "Цена у курорта сильнее зависит от сезона, чем у города",
}
EXPECTED_SIGN = {"H1": 1, "H2": 1, "H3": -1, "H4": -1, "H5": 1, "H6": 1}
ROBUSTNESS_CHECKS = ("C", "A", "B", "C без Non Refund")
BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_SEED = 0


def _rate_diff(df: pd.DataFrame, flag: pd.Series) -> StatResult:
    """z-тест: доля отмен при выполненном условии против доли при невыполненном."""
    yes, no = df.loc[flag, "is_canceled"], df.loc[~flag, "is_canceled"]
    return two_proportion_ztest(int(yes.sum()), len(yes), int(no.sum()), len(no))


def segments(df: pd.DataFrame) -> pd.Series:
    """Сегмент продаж; малые сегменты (Aviation, Complementary) объединены в «Прочие»."""
    return df["market_segment"].where(~df["market_segment"].isin(SMALL_SEGMENTS), OTHER_SEGMENT)


def h1_lead_time(df: pd.DataFrame) -> StatResult:
    """H1: связь корзины срока брони и отмены (χ², V Крамера)."""
    return chi2_independence(lead_time_bins(df["lead_time"]), df["is_canceled"])


def h1_trend(df: pd.DataFrame) -> StatResult:
    """H1, дополнительно: монотонный тренд отмен по сроку брони (Спирмен)."""
    return spearman(df["lead_time"], df["is_canceled"])


def h2_segments(df: pd.DataFrame) -> StatResult:
    """H2: связь сегмента продаж и отмены (χ², V Крамера)."""
    return chi2_independence(segments(df), df["is_canceled"])


def h2_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """H2, попарно: разница долей отмен для каждой пары сегментов с поправкой Холма."""
    seg = segments(df)
    rows = []
    for seg_a, seg_b in combinations(sorted(seg.unique()), 2):
        pair = seg.isin([seg_a, seg_b])
        res = _rate_diff(df[pair], seg[pair] == seg_a)
        rows.append({"segment_a": seg_a, "segment_b": seg_b, "diff": res.effect, "p_value": res.p_value})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    return out


def h3_special_requests(df: pd.DataFrame) -> StatResult:
    """H3: доля отмен у броней со спецзапросами минус без них."""
    return _rate_diff(df, df["total_of_special_requests"] >= 1)


def h4_repeated_guest(df: pd.DataFrame) -> StatResult:
    """H4: доля отмен у повторных гостей минус у новых."""
    return _rate_diff(df, df["is_repeated_guest"] == 1)


def h5_city_hotel(df: pd.DataFrame) -> StatResult:
    """H5: доля отмен городского отеля минус курортного."""
    return _rate_diff(df, df["hotel"] == "City Hotel")


def h6_by_hotel(df: pd.DataFrame) -> dict[str, StatResult]:
    """H6, по отелям: Краскел — Уоллис цены по месяцам заезда без бесплатных броней."""
    paid = df[df["is_free"] == 0]
    return {str(hotel): kruskal_wallis(group["adr"], group["arrival_month"]) for hotel, group in paid.groupby("hotel")}


def h6_bootstrap_ci(df: pd.DataFrame, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    """95 % перцентильный бутстреп-ДИ для ε² курорта минус ε² города.

    Строки платных броней пересэмплируются с возвращением отдельно внутри каждого отеля.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paid = df[df["is_free"] == 0]
    hotels = {str(hotel): group[["adr", "arrival_month"]].to_numpy() for hotel, group in paid.groupby("hotel")}
    diffs = []
    for _ in range(resamples):
        effect = {}
        for hotel, rows in hotels.items():
            sample = rows[rng.integers(0, len(rows), len(rows))]
            effect[hotel] = kruskal_wallis(pd.Series(sample[:, 0]), pd.Series(sample[:, 1])).effect
        diffs.append(effect["Resort Hotel"] - effect["City Hotel"])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def h6_seasonality(df: pd.DataFrame) -> StatResult:
    """H6: сезонность цены у курорта сильнее, чем у города.

    Статистика — H курорта; p-value — наибольший из двух отелей (сезонность значима у обоих);
    эффект — ε² курорта минус ε² города (> 0 — у курорта сильнее) с бутстреп-ДИ этой разницы.
    """
    by_hotel = h6_by_hotel(df)
    resort, city = by_hotel["Resort Hotel"], by_hotel["City Hotel"]
    return StatResult(
        resort.test, resort.statistic, max(resort.p_value, city.p_value), "ε² курорт − ε² город",
        resort.effect - city.effect, *h6_bootstrap_ci(df),
    )


_TESTS = {
    "H1": h1_lead_time, "H2": h2_segments, "H3": h3_special_requests,
    "H4": h4_repeated_guest, "H5": h5_city_hotel, "H6": h6_seasonality,
}


def reject_h0(p_holm: pd.Series, effect: pd.Series, expected_sign: pd.Series,
              ci_low: pd.Series, ci_high: pd.Series) -> pd.Series:
    """Решение «отвергаем H0»: p с поправкой < ALPHA, знак эффекта ожидаемый и ДИ эффекта не накрывает 0.

    Если ДИ не посчитан (NaN), решение принимается по p и знаку.
    """
    ci_missing = ci_low.isna() | ci_high.isna()
    ci_excludes_zero = (ci_low > 0) | (ci_high < 0)
    return (p_holm < ALPHA) & (np.sign(effect) == expected_sign) & (ci_missing | ci_excludes_zero)


def hypotheses_table(df: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица H1–H6: тесты, эффекты, поправка Холма и решение по каждой гипотезе."""
    rows = [{"hypothesis": key, "statement": STATEMENTS[key], **asdict(test(df))} for key, test in _TESTS.items()]
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    expected = out["hypothesis"].map(EXPECTED_SIGN)
    out["reject_h0"] = reject_h0(out["p_holm"], out["effect"], expected, out["ci_low"], out["ci_high"])
    return out


def robustness(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Решения по H1–H6 на вариантах C, A, B и на C без броней «без возврата»; устойчиво — если везде одинаково."""
    frames = {variant: build_dataset(df_raw, variant)[0] for variant in VARIANTS}
    frames["C без Non Refund"] = frames["C"][frames["C"]["deposit_type"] != NON_REFUND]
    decisions = pd.DataFrame({
        name: hypotheses_table(frames[name]).set_index("hypothesis")["reject_h0"] for name in ROBUSTNESS_CHECKS
    })
    decisions["robust"] = decisions.eq(decisions["C"], axis=0).all(axis=1)
    return decisions


def save_hypotheses(table: pd.DataFrame, decisions: pd.DataFrame, path: Path = HYPOTHESES_PATH) -> None:
    """Сохранить сводную таблицу гипотез с отметкой устойчивости в CSV."""
    out = table.merge(decisions[["robust"]], left_on="hypothesis", right_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
