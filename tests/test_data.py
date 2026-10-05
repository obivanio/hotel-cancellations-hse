"""Проверка загрузки и соответствия тестовой фикстуры реальным данным."""
from src.data import load_raw


def test_base_row_matches_raw_columns(make_raw):
    """Колонки эталонной строки совпадают с колонками исходного датасета (порядок тоже)."""
    assert list(make_raw([{}]).columns) == list(load_raw().columns)


def test_raw_shape():
    """Исходный датасет: 119 390 броней × 32 колонки."""
    assert load_raw().shape == (119390, 32)
