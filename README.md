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
src/          код: загрузка (data), очистка (cleaning, cleaning_log), признаки (features), сборка (pipeline),
              разведочный анализ (eda), статистика (stats), гипотезы (hypotheses), графики (plots),
              модель: сплит (split), признаки модели (model_features), правила (baselines), модели (models),
              оценка и порог (evaluation); экономика: правила предоплаты (policies), овербукинг (overbooking),
              интерактивный отчёт (report), дашборд: расчёты (dashboard), сборка (html_dashboard),
              пояснения к схемам (archify_explain); шаблон, стили и логика страницы — src/dashboard_assets/
diagrams/     исходники схем archify 19 и 20, русский интерфейс просмотрщика, пояснения шагов
static/fonts/ шрифты дашборда Golos Text и Unbounded (лицензия OFL)
tests/        тесты pytest
notebooks/    ноутбуки с ходом работы и выводами (01 — загрузка и очистка, 02 — анализ и гипотезы, 03 — модель,
              04 — экономика предоплаты и овербукинга)
data/raw/     исходные данные (скачиваются автоматически, не хранятся в git)
data/processed/  очищенный датасет (создаётся ноутбуком, не хранится в git)
models/       обученная модель CatBoost
reports/      артефакты: журнал очистки, таблица гипотез, метрики моделей, таблицы правил и овербукинга,
              интерактивный отчёт report.html и дашборд dashboard.html (открываются в браузере без Python),
              графики и схемы (figures/)
```

## Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
jupyter notebook notebooks/
```

## Дашборд

![Дашборд](reports/figures/21_dashboard.png)

Откройте `reports/dashboard.html` двойным щелчком — нужен только браузер (Chrome или Safari), интернет и Python не
нужны. Вверху выбираются отель, сегменты, месяцы заезда и допущения о деньгах; черту на очереди броней можно тянуть
мышью или двигать стрелками. Схемы внизу страницы берутся из соседней папки `reports/figures/`, поэтому пересылать
нужно всю папку `reports`.

Пересобрать дашборд после изменения данных или кода:

```bash
python -m src.html_dashboard        # reports/dashboard.html из reports/scored_test.csv, overbooking.csv, policies.csv
python -m src.archify_explain       # пояснения шагов в схемах 19 и 20 (после пересборки схем archify)
```

Тест сверки расчётов браузера с Python (`tests/test_dashboard_js.py`) запускает Node.js; без Node он пропускается.

## Проверка кода

```bash
pytest
flake8 src tests
pylint src tests
nbqa flake8 notebooks/
```

Интеграционные тесты (`test_real_data_*`) работают на настоящем датасете: при первом запуске он
скачивается автоматически, поэтому нужен доступ в интернет.
