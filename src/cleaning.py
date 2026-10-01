"""Правила очистки броней: каждое правило пишет в журнал, что сделало и почему."""
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.cleaning_log import COLLAPSE, MODIFY, REMOVE, CleaningLog, LogEntry

ADR_ERROR_THRESHOLD = 5000
MAX_ADULTS = 10
MAX_CHILDREN_OR_BABIES = 8
RARE_AGENT_MIN_BOOKINGS = 100
VARIANTS = ("A", "B", "C")


@dataclass(frozen=True)
class RemovalRule:
    """Правило удаления строк: условие отбора и обоснование."""

    step: str
    rule: str
    reason: str
    condition: Callable[[pd.DataFrame], pd.Series]


def apply_removal(df: pd.DataFrame, rule: RemovalRule, log: CleaningLog) -> pd.DataFrame:
    """Удалить строки, подходящие под правило, и записать это в журнал."""
    mask = rule.condition(df)
    out = df.loc[~mask]
    log.add(LogEntry(rule.step, REMOVE, rule.rule, len(df), len(out), int(mask.sum()), rule.reason))
    return out


def _total_nights(df: pd.DataFrame) -> pd.Series:
    """Число ночей брони: будни + выходные."""
    return df["stays_in_weekend_nights"] + df["stays_in_week_nights"]


def _total_guests(df: pd.DataFrame) -> pd.Series:
    """Число гостей брони: взрослые + дети + младенцы."""
    return df["adults"] + df["children"] + df["babies"]


REMOVAL_RULES = (
    RemovalRule(
        "1",
        "Отрицательная цена за ночь",
        "Цена не может быть отрицательной — ошибка ввода (в данных 1 такая бронь, −6,38 €).",
        lambda d: d["adr"] < 0,
    ),
    RemovalRule(
        "2",
        f"Цена за ночь ≥ {ADR_ERROR_THRESHOLD} €",
        "Единственная бронь за 5 400 € в 10 раз дороже следующей (510 €), при этом 2 взрослых на 1 ночь "
        "— ошибка ввода.",
        lambda d: d["adr"] >= ADR_ERROR_THRESHOLD,
    ),
    RemovalRule(
        "3",
        f"Детей или младенцев > {MAX_CHILDREN_OR_BABIES}, взрослых > {MAX_ADULTS}",
        "Столько гостей в один номер не поселить: 10 детей, 9–10 младенцев, 20–55 взрослых — "
        "ошибки ввода или служебные записи (около 15 броней).",
        lambda d: (
            (d["children"] > MAX_CHILDREN_OR_BABIES)
            | (d["babies"] > MAX_CHILDREN_OR_BABIES)
            | (d["adults"] > MAX_ADULTS)
        ),
    ),
    RemovalRule(
        "4",
        "Бронь на 0 ночей",
        "715 броней без единой ночи, все с ценой 0 — это посещения на день, а не проживание в номере.",
        lambda d: _total_nights(d) == 0,
    ),
    RemovalRule(
        "5",
        "Бронь без гостей",
        "180 броней без взрослых, детей и младенцев не описывают реальное проживание.",
        lambda d: _total_guests(d) == 0,
    ),
    RemovalRule(
        "6",
        "Сегмент или канал продаж Undefined",
        "Категория без смысла и слишком мала для отдельной группы: 2 брони по сегменту, 5 по каналу.",
        lambda d: (d["market_segment"] == "Undefined") | (d["distribution_channel"] == "Undefined"),
    ),
)


def fix_children_type(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Шаг 10: пропуски в children заменить на 0 и привести колонку к целому типу."""
    missing = int(df["children"].isna().sum())
    out = df.assign(children=df["children"].fillna(0).astype(int))
    log.add(LogEntry(
        "10", MODIFY, "children: пропуск → 0, тип int", len(df), len(out), missing,
        "Пропусков 4 из 119 тыс.; 0 — самое частое значение (93 %); дробный тип был только из-за пропусков.",
    ))
    return out


def merge_undefined_meal(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Шаг 7: питание Undefined объединить с SC."""
    affected = int((df["meal"] == "Undefined").sum())
    out = df.assign(meal=df["meal"].replace("Undefined", "SC"))
    log.add(LogEntry(
        "7", MODIFY, "Питание Undefined → SC", len(df), len(out), affected,
        "По документации датасета (Antonio et al., 2019) Undefined и SC означают одно: без пакета питания.",
    ))
    return out


def fill_unknown_country(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Шаг 8: пустую страну заменить категорией unknown."""
    affected = int(df["country"].isna().sum())
    out = df.assign(country=df["country"].fillna("unknown"))
    log.add(LogEntry(
        "8", MODIFY, "Пустая страна → unknown", len(df), len(out), affected,
        "Пропуск информативен — страну часто уточняют при заезде; удаление стоило бы 488 броней.",
    ))
    return out


def drop_week_number(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Шаг 11: удалить колонку номера недели — она противоречит дате заезда."""
    out = df.drop(columns="arrival_date_week_number")
    log.add(LogEntry(
        "11", MODIFY, "Удалена колонка arrival_date_week_number", len(df), len(out), len(df),
        "Номер недели не совпадает с календарной неделей даты заезда в 54 % строк; неделю считаем из даты.",
    ))
    return out


def _with_key_and_rooms(df: pd.DataFrame, n_rooms) -> pd.DataFrame:
    """Добавить booking_key первой колонкой и n_rooms последней."""
    out = df.reset_index(drop=True).assign(n_rooms=n_rooms)
    out.insert(0, "booking_key", range(len(out)))
    return out


def collapse_duplicates(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Шаг 12 (вариант C): одинаковые строки склеить в одну бронь, число номеров — в n_rooms."""
    grouped = df.groupby(list(df.columns), dropna=False, sort=False).size()
    out = _with_key_and_rooms(grouped.index.to_frame(index=False), grouped.to_numpy())
    log.add(LogEntry(
        "12", COLLAPSE, "Одинаковые строки → одна бронь с n_rooms", len(df), len(out), len(df) - len(out),
        "ID брони удалён при обезличивании, поэтому одинаковые строки — скорее номера одной групповой брони "
        "(75 % строк сегмента «Группы» — копии). Удаление занизило бы отмены (37 % → 27,5 %) и выручку, "
        "сохранение как есть сделало бы зависимые наблюдения независимыми. Тесты и модель — по броням, "
        "деньги — с весом n_rooms.",
    ))
    return out


def drop_duplicate_rows(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Шаг 12, вариант A (для проверки устойчивости): удалить повторяющиеся строки."""
    out = _with_key_and_rooms(df.drop_duplicates(), 1)
    log.add(LogEntry(
        "12A", REMOVE, "Вариант A: повторяющиеся строки удалены", len(df), len(out), len(df) - len(out),
        "Вариант для проверки устойчивости выводов: дубли считаются ошибочными копиями.",
    ))
    return out


def keep_duplicate_rows(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Шаг 12, вариант B (для проверки устойчивости): оставить все строки как есть."""
    out = _with_key_and_rooms(df, 1)
    log.add(LogEntry(
        "12B", MODIFY, "Вариант B: дубли оставлены", len(df), len(out), 0,
        "Вариант для проверки устойчивости выводов: каждая строка считается отдельной бронью.",
    ))
    return out


def group_rare_agents(
    df: pd.DataFrame, log: CleaningLog, min_bookings: int = RARE_AGENT_MIN_BOOKINGS
) -> pd.DataFrame:
    """Шаг 9 (после склейки): агентов с числом броней меньше порога объединить в other."""
    counts = df["agent"].value_counts()
    rare = df["agent"].isin(counts.index[counts < min_bookings])
    agent_id = df["agent"].astype("Int64").astype("string")
    out = df.assign(agent=agent_id.mask(rare, "other"), agent_id=agent_id)
    log.add(LogEntry(
        "9", MODIFY, f"Агенты с < {min_bookings} бронями → other", len(df), len(out), int(rare.sum()),
        "Пусто в agent/company = «без агента / не от компании», а не пропуск: оставляем, флаги — в признаках. "
        "Редкие агенты дают шум, поэтому объединяются; считаем после склейки, чтобы не склеить разные брони.",
    ))
    return out


def _prepare(df_raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Общие шаги всех вариантов: тип children, удаления 1–6, нормализация 7, 8, 11."""
    df = fix_children_type(df_raw, log)
    for rule in REMOVAL_RULES:
        df = apply_removal(df, rule, log)
    df = merge_undefined_meal(df, log)
    df = fill_unknown_country(df, log)
    return drop_week_number(df, log)


_DUPLICATE_HANDLERS = {"A": drop_duplicate_rows, "B": keep_duplicate_rows, "C": collapse_duplicates}


def make_variant(df_raw: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, CleaningLog]:
    """Очистить сырые данные с заданной обработкой дублей: A — удалить, B — оставить, C — склеить."""
    if variant not in VARIANTS:
        raise ValueError(f"Неизвестный вариант {variant!r}, ожидается один из {VARIANTS}")
    log = CleaningLog()
    df = _prepare(df_raw, log)
    df = _DUPLICATE_HANDLERS[variant](df, log)
    return group_rare_agents(df, log), log


def clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """Основная очистка — вариант C (склейка дублей в брони с n_rooms)."""
    return make_variant(df_raw, "C")
