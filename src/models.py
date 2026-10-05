"""Обучение моделей риска отмены: логистическая регрессия, HistGradientBoosting, CatBoost; калибровка."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.model_features import LOG_COLUMNS, categorical_columns
from src.pipeline import ROOT

RANDOM_STATE = 0
MODEL_PATH = ROOT / "models" / "catboost.cbm"
CATBOOST_ITERATIONS = 2000
CATBOOST_EARLY_STOPPING = 100
Data = tuple[pd.DataFrame, pd.Series]
CATBOOST_GRID = [{"depth": depth, "learning_rate": rate} for depth in (4, 6, 8) for rate in (0.05, 0.1)]


def fit_logistic(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Логистическая регрессия: log1p для длинных хвостов, стандартизация числовых, one-hot категорий."""
    categorical = categorical_columns(X)
    numeric = [column for column in X.columns if column not in categorical]
    plain = [column for column in numeric if column not in LOG_COLUMNS]
    transform = ColumnTransformer([
        ("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()),
         LOG_COLUMNS),
        ("num", StandardScaler(), plain),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = make_pipeline(transform, LogisticRegression(max_iter=2000))
    return model.fit(X, y)


def fit_hist_gb(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    """HistGradientBoosting из scikit-learn с настройками по умолчанию и родной обработкой категорий."""
    model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=RANDOM_STATE)
    return model.fit(as_category(X), y)


def as_category(X: pd.DataFrame) -> pd.DataFrame:
    """Строковые колонки → тип category (нужен HistGradientBoosting)."""
    return X.astype({column: "category" for column in categorical_columns(X)})


def fit_catboost(train: Data, valid: Data, depth: int = 6, learning_rate: float = 0.1) -> CatBoostClassifier:
    """CatBoost с ранней остановкой по валидации (число деревьев подбирается автоматически).

    train и valid — пары (X, y) из model_frame.
    """
    model = CatBoostClassifier(
        iterations=CATBOOST_ITERATIONS, depth=depth, learning_rate=learning_rate, eval_metric="Logloss",
        early_stopping_rounds=CATBOOST_EARLY_STOPPING, random_seed=RANDOM_STATE, verbose=False,
        allow_writing_files=False,
    )
    X_train, y_train = train
    return model.fit(X_train, y_train, cat_features=categorical_columns(X_train), eval_set=valid, use_best_model=True)


def tune_catboost(train: Data, valid: Data) -> tuple[CatBoostClassifier, pd.DataFrame]:
    """Перебор глубины и скорости обучения CatBoost; лучшая модель — по PR-AUC на валидации."""
    X_valid, y_valid = valid
    rows, best, best_score = [], None, -1.0
    for params in CATBOOST_GRID:
        model = fit_catboost(train, valid, **params)
        score = average_precision_score(y_valid, model.predict_proba(X_valid)[:, 1])
        rows.append({**params, "trees": model.tree_count_, "pr_auc": score})
        if score > best_score:
            best, best_score = model, score
    return best, pd.DataFrame(rows)


def fit_calibrator(proba: np.ndarray, y: pd.Series) -> IsotonicRegression:
    """Изотоническая калибровка: сырая вероятность модели → наблюдаемая доля отмен."""
    return IsotonicRegression(out_of_bounds="clip").fit(proba, y)


def save_catboost(model: CatBoostClassifier, path: Path = MODEL_PATH) -> Path:
    """Сохранить модель CatBoost в файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path


def load_catboost(path: Path = MODEL_PATH) -> CatBoostClassifier:
    """Загрузить модель CatBoost из файла."""
    return CatBoostClassifier().load_model(str(path))


def odds_ratios(model: Pipeline) -> pd.Series:
    """Отношения шансов логистической регрессии: во сколько раз меняются шансы отмены.

    Для числового признака — при росте на одно стандартное отклонение (для lead_time и adr — log1p),
    для категории — по сравнению со средним уровнем; по убыванию силы влияния.
    """
    names = model[0].get_feature_names_out()
    odds = pd.Series(np.exp(model[-1].coef_[0]), index=names)
    return odds.reindex(np.log(odds).abs().sort_values(ascending=False).index)
