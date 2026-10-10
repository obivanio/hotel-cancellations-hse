"""Дашборд для бизнеса: с каких броней брать предоплату и насколько овербукировать даты (streamlit run app.py)."""
import pandas as pd
import streamlit as st

from src.dashboard import (
    HOTELS, SCORED_TEST_PATH, SCORED_VALID_PATH, count, dates_slice, display_rules, fig_heatmap, fig_share_curves,
    fig_weeks, filter_frame, key_numbers, money, overbooking_summary, percent, risk_tables, rules_table,
    share_curves, to_scored,
)
from src.evaluation import Assumptions
from src.overbooking import OVERBOOK_CONFIDENCE, OVERBOOKING_PATH
from src.plots import FIGURES_DIR
from src.policies import best_lead_days
from src.report import MONTH_NAMES

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


def comma(value: float, digits: int = 2) -> str:
    """Число с десятичной запятой."""
    return f"{value:.{digits}f}".replace(".", ",")


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


def cards(numbers: dict) -> None:
    """Ключевые цифры среза и порог модели."""
    columns = st.columns(5)
    columns[0].metric("Без предоплаты", money(numbers["baseline"]), help="Потери от отмен, если не брать предоплату.")
    columns[1].metric("Предоплата со всех", money(numbers["everyone"]),
                      help="Потери, если брать предоплату с каждой брони.")
    columns[2].metric("Предоплата по модели", money(numbers["model"]),
                      help="Потери, если брать предоплату только с рискованных броней.")
    columns[3].metric("Модель дешевле «со всех»", money(numbers["saving"]))
    columns[4].metric("Задето надёжных гостей", percent(numbers["touched_reliable"]),
                      help="Доля гостей, которые и так бы приехали, но с них по модели взяли бы предоплату.")
    st.caption(f"Модель берёт предоплату с броней, у которых риск отмены не ниже {comma(numbers['threshold'])} "
               f"({percent(numbers['prepay_share'])} выбранных броней). Порог зависит от допущений слева.")


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
    st.plotly_chart(fig_share_curves(curves, share), width="stretch")
    lead_days = lead_days_for(assumptions.unsold_share, assumptions.loss_prob)
    table = rules_table(frame, lead_days, share, assumptions)
    top = table.iloc[-1]
    by_lead = curves[(curves["method"] == "срок брони") & curves["share"].round(2).eq(round(share, 2))]
    st.markdown(
        f"При предоплате с {percent(share)} самых рискованных броней потери — {money(top['total'])}, задето "
        f"{percent(top['touched_reliable'])} надёжных гостей, поймано {percent(top['caught_cancels'])} отмен."
        + (f" Если взять столько же броней по сроку, потери — {money(by_lead['total'].iloc[0])}."
           if not by_lead.empty else ""))
    st.subheader("Пять правил для выбранных броней")
    st.dataframe(display_rules(table), hide_index=True, width="stretch")
    st.caption(f"N для правила по сроку подобран на валидации при текущих допущениях: {lead_days} дн. Порог модели "
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
    st.plotly_chart(fig_weeks(part, filters["hotel"]), width="stretch")
    summary = overbooking_summary(dates, filters["hotel"], filters["months"])
    model, bench = summary["модель"], summary["бенчмарк"]
    columns = st.columns(4)
    columns[0].metric("Перепродано номеров", count(model["resold"]))
    columns[1].metric("Дат, где номера не хватило", count(model["short_dates"]))
    columns[2].metric("Гостей без номера", count(model["walked"]))
    columns[3].metric("Итог в деньгах", money(model["net"]))
    st.markdown(
        f"Осторожная оценка — число отмен, которое наступит с вероятностью не ниже {percent(OVERBOOK_CONFIDENCE)}. "
        f"Если продавать столько номеров сверх броней, модель перепродаёт {count(model['resold'])} номеров, "
        f"бенчмарк — {count(bench['resold'])}.\n\n"
        "Гость без номера стоит отелю одну стоимость его брони. Отмены в группах происходят вместе, поэтому на деле "
        "номера не хватает чаще, чем обещает оценка: овербукировать стоит осторожно и по каждому отелю отдельно.")


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
    left.plotly_chart(fig_heatmap(tables["risk"], tables["bookings"], "Средний риск по модели, %"), width="stretch")
    right.plotly_chart(fig_heatmap(tables["cancel"], tables["bookings"], "Доля отмен на деле, %"), width="stretch")


def tab_model() -> None:
    """Вкладка «Как собрана модель»: схемы 20 и 19."""
    st.markdown(
        "Путь модели от исходных данных до порога предоплаты: очистка, признаки, известные при бронировании, "
        "деление по датам, выбор модели на валидации и одна проверка на тесте.")
    st.image(str(FIGURES_DIR / "20_model_full_pipeline.png"))
    st.markdown(
        "Как правило работает для одной брони: модель оценивает риск отмены, и если он не ниже порога, отель берёт "
        "предоплату. Цифры на схеме — при допущениях 50 % и 10 %; в дашборде они пересчитываются по ползункам.")
    st.image(str(FIGURES_DIR / "19_prepayment_decision.png"))


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
    """Раскладка дашборда: фильтры, ключевые цифры, вкладки."""
    st.set_page_config(page_title="Отмены броней: предоплата и овербукинг", layout="wide")
    tables = load_tables()
    filters = sidebar(tables["test"])
    frame = filter_frame(tables["test"], filters["hotel"], filters["segments"], filters["months"])
    st.title("Отмены броней: с кого брать предоплату и насколько овербукировать")
    st.markdown(
        "Брони с заездом в мае–августе 2017 года, два отеля. Риск отмены каждой брони оценила модель, обученная на "
        "более ранних бронях.\n\n"
        f"Выбрано броней: {count(len(frame))}. Все цифры пересчитываются по фильтрам и допущениям слева.")
    numbers = key_numbers(frame, filters["assumptions"])
    if frame.empty:
        st.warning(EMPTY)
    else:
        cards(numbers)
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
