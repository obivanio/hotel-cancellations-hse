# Отмены бронирований в отелях: прогноз и политика предоплаты

Итоговый проект программы ДПО НИУ ВШЭ «Аналитика данных».

**Вопрос:** какие брони отменят и сколько отель сэкономит, если брать предоплату только с
рискованных броней?

## Данные

Hotel Booking Demand — 119 390 броней двух отелей в Португалии (городской и курортный),
заезды с июля 2015 по август 2017, 32 признака.
Источник: Antonio, Almeida, Nunes. *Hotel booking demand datasets*. Data in Brief, 2019.

Датасет скачивается автоматически при первом запуске (`src/data.py`) в `data/raw/`.

## Структура

```
src/          код: загрузка (data), очистка (cleaning, cleaning_log), признаки (features), сборка (pipeline)
tests/        тесты pytest
notebooks/    ноутбуки с ходом работы и выводами (01 — загрузка и очистка)
data/raw/     исходные данные (скачиваются автоматически, не хранятся в git)
reports/      артефакты: журнал очистки, графики, таблицы
```

## Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
jupyter notebook notebooks/
```

## Проверка кода

```bash
pytest
flake8 src tests
pylint src tests
nbqa flake8 notebooks/
```
