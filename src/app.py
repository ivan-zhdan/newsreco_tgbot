import joblib
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from catboost import CatBoostClassifier
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Вычисление Абсолютных Путей (ИСПРАВЛЕНИЕ ОШИБКИ) ---
# Получаем путь к папке, где лежит текущий файл app.py (т.е. к папке src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Поднимаемся на уровень выше в корень проекта
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Формируем абсолютные пути к артефактам в папке models/ в корне проекта
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'catboost_ranker.cbm')
STATS_PATH = os.path.join(PROJECT_ROOT, 'models', 'item_stats.pkl')

# --- Глобальные переменные для артефактов ML ---
ml_artifacts = {}


# --- Lifespan Handler (ЗАМЕНА ДЛЯ ON_EVENT) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекст-менеджер для управления жизненным циклом приложения.
    Код до yield выполняется при старте, после yield — при остановке.
    """
    # [STARTUP LOGIC] Загружаем артефакты при старте
    logger.info(f"Loading ML artifacts from:")
    logger.info(f"  Model: {MODEL_PATH}")
    logger.info(f"  Stats: {STATS_PATH}")

    try:
        # Убедимся, что файлы существуют перед загрузкой
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not os.path.exists(STATS_PATH):
            raise FileNotFoundError(f"Stats file not found at {STATS_PATH}")

        # Загрузка модели CatBoost
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)

        # Загрузка статистик
        stats = joblib.load(STATS_PATH)

        # Сохраняем в глобальный словарь состояний
        ml_artifacts['model'] = model
        ml_artifacts['item_clicks'] = stats['clicks']
        ml_artifacts['item_impressions'] = stats['impressions']

        logger.info("✅ ML Artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {e}")
        # В проде лучше остановить запуск, если модель критична.
        # Для пет-проекта оставим словарь пустым.

    yield  # --- Приложение работает и принимает запросы ---

    # [SHUTDOWN LOGIC] Код здесь выполнится при остановке приложения
    logger.info("Shutting down API, clearing artifacts...")
    ml_artifacts.clear()


# Инициализация FastAPI с указанием lifespan
app = FastAPI(title="Recommendation System API", lifespan=lifespan)


# --- Модели данных (Pydantic) ---
class CandidateNews(BaseModel):
    news_id: str
    category: str  # Категория из news.tsv для продвинутых фичей


class RecommendRequest(BaseModel):
    user_id: str
    history: List[str]  # История кликов пользователя
    candidates: List[CandidateNews]  # Список кандидатов с категориями


class RecommendationResponse(BaseModel):
    user_id: str
    ranked_news_ids: List[str]


# --- Утилитная функция для признаков (использует ml_artifacts) ---
def generate_features_on_the_fly(request: RecommendRequest) -> pd.DataFrame:
    """Генерирует датафрейм признаков для пар user-candidate."""
    history_len = len(request.history)
    item_clicks = ml_artifacts.get('item_clicks', {})
    item_impressions = ml_artifacts.get('item_impressions', {})

    features_list = []

    for pos, candidate in enumerate(request.candidates):
        news_id = candidate.news_id

        # Рассчитываем признаки на основе загруженных статистик
        clicks = item_clicks.get(news_id, 0)
        impressions = item_impressions.get(news_id, 0)
        ctr = clicks / impressions if impressions > 0 else 0.0

        feat_row = {
            # Колонки должны соответствовать тем, на которых училась модель!
            'position_in_impression': pos,
            'history_len': history_len,
            'train_clicks': clicks,
            'train_impressions': impressions,
            'train_ctr': ctr,
            'item_category': candidate.category,
            # Заглушки для признаков, требующих более сложной логики (см. ноутбук)
            'user_fav_category': 'unknown',
            'is_fav_category_match': 0,
            'category_preference_score': 0.0
        }
        features_list.append(feat_row)

    return pd.DataFrame(features_list)


# --- Эндпоинты ---
@app.get("/health")
def health_check():
    """Простой чек работоспособности и загрузки модели."""
    model_loaded = 'model' in ml_artifacts
    status = "ok" if model_loaded else "error_no_model"
    return {"status": status, "model_loaded": model_loaded}

@app.get("/")
def read_root():
    return {"status": "online", "message": "RecSys API is running"}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendRequest):
    """Принимает историю и кандидатов, возвращает Top-5 отранжированных новостей."""
    model = ml_artifacts.get('model')
    if model is None:
        raise HTTPException(status_code=503, detail="ML Model not available. Check logs.")

    if not request.candidates:
        return RecommendationResponse(user_id=request.user_id, ranked_news_ids=[])

    try:
        # 1. Генерация признаков
        X_predict = generate_features_on_the_fly(request)

        # 2. Строгий порядок колонок для CatBoost
        feature_cols = [
            'position_in_impression', 'history_len', 'train_clicks', 'train_impressions', 'train_ctr',
            'is_fav_category_match', 'category_preference_score', 'item_category', 'user_fav_category'
        ]
        # Гарантируем наличие и порядок всех колонок
        for col in feature_cols:
            if col not in X_predict.columns:
                X_predict[col] = 0  # Или другое дефолтное значение

        X_predict = X_predict[feature_cols]

        # 3. Инференс (получаем вероятность клика)
        scores = model.predict_proba(X_predict)[:, 1]

        # 4. Ранжирование и выбор Top-5
        candidate_ids = [c.news_id for c in request.candidates]
        # Сортируем ID кандидатов по убыванию скора
        ranked_results = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        ranked_news_ids = [item[0] for item in ranked_results[:5]]

        return RecommendationResponse(user_id=request.user_id, ranked_news_ids=ranked_news_ids)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")


# Запуск: python src/app.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)