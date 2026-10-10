"""Дашборд для бизнеса: с каких броней брать предоплату и насколько овербукировать даты (streamlit run app.py)."""
import pandas as pd
import streamlit as st

from src.dashboard import (
    HOTELS, SCORED_TEST_PATH, SCORED_VALID_PATH, comma, count, dates_slice, display_rules, fig_heatmap,
    fig_share_curves, fig_weeks, filter_frame, key_numbers, lead_total_at_share, money, overbooking_summary,
    percent, risk_tables, rules_table, saving_card, share_curves, to_scored, top_share_row, wall_counts, wall_html,
)
from src.evaluation import Assumptions
from src.overbooking import OVERBOOK_CONFIDENCE, OVERBOOKING_PATH
from src.plots import FIGURES_DIR
from src.policies import best_lead_days
from src.report import MONTH_NAMES
from src.theme import CSS, OCHRE

TABS = ["Правила предоплаты", "Овербукинг", "Риск по сегментам", "Как собрана модель", "Допущения и ограничения"]
EMPTY = "В выбранном срезе нет броней. Измените фильтры слева: например, такого сегмента нет в этом отеле."


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    """Готовые таблицы шага 4: вероятности броней теста и валидации, отмены по датам заезда."""
    return {"test": pd.read_csv(SCORED_TEST_PATH), "valid": pd.read_csv(SCORED_VALID_PATH),
            "dates": pd.read_csv(OVERBOOKING_PATH, parse_dates=["arrival_date"])}


@st.cache_data
def lead_days_for(unsold_share: float, loss_prob: float) -> int:
    """N для правила «срок брони > N дней», подобранный на валидации при текущих допущениях (как в ноутбуке 04)."""
    return best_lead_days(to_scored(load_tables()["valid"]), Assumptions(unsold_share, loss_prob))


def sidebar(test: pd.DataFrame) -> dict:
    """Фильтры и допущения в боковой панели."""
    st.sidebar.header("Фильтры")
    hotel = st.sidebar.radio("Отель", list(HOTELS), key="hotel")
    segments = sorted(test["market_segment"].unique())
    months = sorted(test["arrival_month"].unique())
    chosen_segments = st.sidebar.multiselect("Сегмент продаж", segments, default=segments, key="segments")
    chosen_months = st.sidebar.multiselect("Месяц заезда (2017)", months, default=months, key="months",
                                           format_func=MONTH_NAMES.get)
    st.sidebar.header("Допущения о деньгах")
    unsold = st.sidebar.slider("Теряем при пропущенной отмене, % стоимости брони", 30, 70, 50, 5, key="unsold",
                               help="Какую часть стоимости отменённой брони не удаётся вернуть перепродажей номера.")
    loss = st.sidebar.slider("Уходит надёжных гостей из-за предоплаты, %", 5, 20, 10, 1, key="loss",
                             help="Сколько гостей, которые и так бы приехали, откажутся от брони из-за предоплаты.")
    return {"hotel": HOTELS[hotel], "segments": chosen_segments, "months": chosen_months,
            "assumptions": Assumptions(unsold / 100, loss / 100)}


def key_figures(numbers: dict) -> None:
    """Ключевые цифры среза справа от стены: потери трёх вариантов, задетые гости, экономия модели и порог."""
    figures = [
        ("kpi_baseline", "Без предоплаты", money(numbers["baseline"]), "Потери от отмен, если не брать предоплату."),
        ("kpi_everyone", "Со всех", money(numbers["everyone"]), "Потери, если брать предоплату с каждой брони."),
        ("kpi_model", "По модели", money(numbers["model"]),
         "Потери, если брать предоплату только с рискованных броней."),
        ("kpi_touched", "Задето надёжных гостей", percent(numbers["touched_reliable"]),
         "Доля гостей, которые и так бы приехали, но с них по модели взяли бы предоплату."),
    ]
    for row in (figures[:2], figures[2:]):
        for column, (key, label, value, hint) in zip(st.columns(2), row):
            column.container(key=key).metric(label, value, help=hint)
    label, value = saving_card(numbers["saving"])
    st.markdown(f"**{label} {value}.**")
    st.caption(f"Модель берёт предоплату с броней, у которых риск отмены не ниже {comma(numbers['threshold'])}: "
               f"это {percent(numbers['prepay_share'])} выбранных броней. Порог зависит от допущений слева.\n\n"
               "На всех бронях теста главный выигрыш против «без предоплаты» даёт сама предоплата: «со всех» "
               "теряет почти столько же, сколько «по модели». Заслуга модели — чуть меньше потерь и меньше "
               "задетых надёжных гостей.")


def header(frame: pd.DataFrame, filters: dict) -> None:
    """Заголовок, стена из 100 плиток слева и ключевые цифры справа; у пустого среза — только предупреждение."""
    st.title("С кого брать предоплату")
    st.markdown(f"Брони с заездом в мае–августе 2017 года, два отеля в Португалии. Выбрано броней: "
                f"{count(len(frame))}; все цифры пересчитываются по фильтрам и допущениям слева.")
    if frame.empty:
        st.warning(EMPTY)
        return
    wall, figures = st.columns([5, 7], gap="large")
    wall.markdown(wall_html(wall_counts(frame, filters["assumptions"]), filters["hotel"]), unsafe_allow_html=True)
    with figures:
        key_figures(key_numbers(frame, filters["assumptions"]))


def tab_rules(frame: pd.DataFrame, assumptions: Assumptions) -> None:
    """Вкладка «Правила предоплаты»: кривая «доля предоплаты → потери» и таблица пяти правил."""
    st.markdown(
        "Предоплату можно брать с разной доли броней. Чем больше доля, тем меньше потерь от отмен, но тем больше "
        "надёжных гостей, которые и так бы приехали, задевает требование заплатить заранее.\n\n"
        "Сравниваем два способа выбрать брони при одной и той же доле: по риску отмены от модели и по сроку брони "
        "(сначала самые ранние). Ниже линия — меньше потерь.")
    share = st.slider("С какой доли самых рискованных броней брать предоплату, %", 5, 100, 50, 5, key="share") / 100
    if frame.empty:
        st.info(EMPTY)
        return
    curves = share_curves(frame, assumptions)
    st.plotly_chart(fig_share_curves(curves, share), width="stretch", theme=None)
    lead_days = lead_days_for(assumptions.unsold_share, assumptions.loss_prob)
    table = rules_table(frame, lead_days, share, assumptions)
    top = top_share_row(table)
    by_lead = lead_total_at_share(curves, share)
    st.markdown(
        f"При предоплате с {percent(share)} самых рискованных броней потери — {money(top['total'])}, задето "
        f"{percent(top['touched_reliable'])} надёжных гостей, поймано {percent(top['caught_cancels'])} отмен."
        + (f" Если взять столько же броней по сроку, потери — {money(by_lead)}."
           if by_lead is not None else ""))
    st.subheader("Пять правил для выбранных броней")
    st.dataframe(display_rules(table), hide_index=True, width="stretch")
    st.caption(f"N для правила по сроку подобран на всех бронях валидации при текущих допущениях, а не на "
               f"выбранном срезе (на малом срезе подбор случаен): {lead_days} дн. Порог модели "
               "считается по формуле «доля ушедших гостей / (доля потерь при отмене + доля ушедших гостей)».")


def tab_overbooking(dates: pd.DataFrame, filters: dict) -> None:
    """Вкладка «Овербукинг»: отмены по неделям и итог осторожной оценки."""
    st.markdown(
        "Овербукинг — продажа номеров сверх броней в расчёте на отмены. Модель оценивает, сколько отмен будет на "
        "каждую дату заезда; бенчмарк — доля отмен по месяцу и дню недели в прошлом.\n\n"
        "Здесь работают фильтры отеля и месяца; сегмент не учитывается, потому что решение принимается для всей "
        "даты сразу.")
    part = dates_slice(dates, filters["hotel"], filters["months"])
    if part.empty:
        st.info("Нет дат заезда для выбранных месяцев. Выберите хотя бы один месяц слева.")
        return
    st.plotly_chart(fig_weeks(part, filters["hotel"]), width="stretch", theme=None)
    summary = overbooking_summary(dates, filters["hotel"], filters["months"])
    model, bench = summary["модель"], summary["бенчмарк"]
    columns = st.columns(4)
    columns[0].metric("Перепродано номеров", count(model["resold"]))
    columns[1].metric("Дат, где номера не хватило", count(model["short_dates"]))
    columns[2].metric("Гостей без номера", count(model["walked"]))
    columns[3].metric("Итог в деньгах (верхняя оценка)", money(model["net"]),
                      help="Выручка от перепроданных номеров минус стоимость гостей без номера, если каждый "
                           "перепроданный номер удастся продать.")
    st.markdown(
        f"Осторожная оценка — число отмен, которое наступит с вероятностью не ниже {percent(OVERBOOK_CONFIDENCE)}. "
        f"Если продавать столько номеров сверх броней, модель перепродаёт {count(model['resold'])} номеров, "
        f"бенчмарк — {count(bench['resold'])}.\n\n"
        "Итог в деньгах — верхняя оценка: считаем, что на каждый перепроданный номер найдётся покупатель. Если "
        "часть отменённых номеров отель и так перепродаёт, реальная добавка может быть примерно вдвое меньше.\n\n"
        "Сравнение с бенчмарком не вполне равное: бенчмарк берёт доли отмен с более ранних броней, когда отмен было "
        "меньше, поэтому часть разрыва объясняется этим, а не только моделью.\n\n"
        "Гость без номера стоит отелю одну стоимость его брони. Оценка считает отмены независимыми, но в группах они "
        "происходят вместе, поэтому на деле номера не хватает чаще, чем обещает оценка. Вместимости отелей в данных "
        "нет: перепродажа считается сверх забронированного, а не сверх свободных номеров. Овербукировать стоит "
        "осторожно и по каждому отелю отдельно.")


def tab_risk(frame: pd.DataFrame) -> None:
    """Вкладка «Риск по сегментам»: тепловые карты «сегмент × месяц»."""
    st.markdown(
        "Где модель видит риск: средний риск отмены по сегменту продаж и месяцу заезда, рядом — доля броней, "
        "которые отменили на деле.\n\n"
        "Это связь, а не причина: сегмент не вызывает отмену. В клетках с малым числом броней (видно в подсказке) "
        "цифры случайны.")
    if frame.empty:
        st.info(EMPTY)
        return
    tables = risk_tables(frame)
    left, right = st.columns(2)
    left.plotly_chart(fig_heatmap(tables["risk"], tables["bookings"], "Средний риск по модели, %"),
                      width="stretch", theme=None)
    right.plotly_chart(fig_heatmap(tables["cancel"], tables["bookings"], "Доля отмен на деле, %", OCHRE),
                       width="stretch", theme=None)


def diagram(name: str, height: int, alt: str) -> None:
    """Интерактивная схема archify из reports/figures (st.iframe); если HTML-файла нет — картинка PNG."""
    path = FIGURES_DIR / f"{name}.html"
    if not path.exists():
        st.image(str(FIGURES_DIR / f"{name}.png"), caption=alt)
        return
    st.caption("Схему можно двигать мышью и приближать колёсиком мыши или кнопками масштаба на панели схемы. "
               "Подписи кнопок на панели — на английском.")
    st.iframe(path, height=height, alt=alt)


def tab_model() -> None:
    """Вкладка «Как собрана модель»: интерактивные схемы 20 и 19."""
    st.markdown(
        "Путь модели от исходных данных до порога предоплаты: очистка, признаки, известные при бронировании, "
        "деление по датам, выбор модели на валидации и одна проверка на тесте.")
    diagram("20_model_full_pipeline", 720, "Схема: как собрана модель, от исходных данных до порога предоплаты")
    st.markdown(
        "Как правило работает для одной брони: модель оценивает риск отмены, и если он не ниже порога, отель берёт "
        "предоплату. Цифры на схеме — при допущениях 50 % и 10 %; в дашборде они пересчитываются по ползункам.")
    diagram("19_prepayment_decision", 620, "Схема: как модель решает, брать ли предоплату с одной брони")


def tab_assumptions(assumptions: Assumptions) -> None:
    """Вкладка «Допущения и ограничения»."""
    st.markdown(
        f"- При пропущенной отмене отель теряет {percent(assumptions.unsold_share)} стоимости брони: остальное "
        "возвращает перепродажа номера.\n"
        f"- Надёжный гость уходит из-за требования предоплаты с вероятностью {percent(assumptions.loss_prob)}.\n"
        "- Этих цифр нет в данных, поэтому их можно менять ползунками слева; вывод о лучшем правиле стоит "
        "проверять при разных значениях.\n"
        "- Брони «без возврата» уже оплачены: их отмена ничего не стоит отелю, и они не считаются ни задетыми "
        "гостями, ни пойманными отменами.\n"
        "- Деньги предоплаты и овербукинга не складываются: оба решения опираются на одни и те же отмены.\n"
        "- Вместимости отелей в данных нет: перепродажа считается сверх фактически забронированного.\n"
        "- Отмены считаются независимыми; групповые отмены это нарушают.\n"
        "- Проверка — брони с заездом в мае–августе 2017 года, два отеля в Португалии. Это связь, а не причина.")


def main() -> None:
    """Раскладка дашборда: фильтры, заголовок со стеной и ключевыми цифрами, вкладки."""
    st.set_page_config(page_title="С кого брать предоплату", layout="wide")
    st.html(f"<style>{CSS}</style>")
    tables = load_tables()
    filters = sidebar(tables["test"])
    frame = filter_frame(tables["test"], filters["hotel"], filters["segments"], filters["months"])
    header(frame, filters)
    rules, overbooking, risk, model, assumptions = st.tabs(TABS)
    with rules:
        tab_rules(frame, filters["assumptions"])
    with overbooking:
        tab_overbooking(tables["dates"], filters)
    with risk:
        tab_risk(frame)
    with model:
        tab_model()
    with assumptions:
        tab_assumptions(filters["assumptions"])


main()
