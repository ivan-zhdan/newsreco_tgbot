# 🗞️ Personal News Recommendation System (End-to-End ML & MLOps Service)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Latest-yellow.svg)](https://catboost.ai/)
[![Docker Compose](https://img.shields.io/badge/Docker--Compose-Supported-blue)](https://www.docker.com/)

Инженерный ML-сервис персональных рекомендаций новостей на базе двухуровневой архитектуры (Two-Stage Recommendation Architecture) с интеграцией в Telegram, встроенным MLOps-контуром отслеживания дрейфа данных (Data Drift) и пайплайном автоматического переобучения (Quality Gate).

---

## 🏗️ 1. Архитектура системы

Система состоит из двух изолированных сервисов, общающихся по HTTP и объединенных через Docker Compose:

```text
[ Telegram User ]
        │
        ▼ (команда /recommend)
┌─────────────────────────────────┐
│     Telegram Bot (Worker)       │  <-- Aiogram 3.x (Async polling)
└───────────────┬─────────────────┘
                │
                ▼ POST /recommend (JSON payload)
┌─────────────────────────────────┐
│     FastAPI ML Inference API    │  <-- Uvicorn (Port 8000)
│   (On-the-fly Feature Eng.)     │
└───────────────┬─────────────────┘
                │
                ▼ (Inference score)
┌─────────────────────────────────┐
│      CatBoostRanker Model       │  <-- Артефакты: catboost_ranker.cbm, item_stats.pkl
└─────────────────────────────────┘
```

## 2. Бизнес-контекст и Постановка задачи

**Цель:** Повысить вовлеченность пользователей (CTR и дочитываемость) за счет персонального ранжирования новостного потока.

**Специфика данных (MIND Dataset):**
* **Временные смещения (Time drift):** новости теряют актуальность через 24–48 часов.
* **Popularity Bias:** явный перекос в сторону топ-новостей.

**Валидация:** Time-based split (обучение на прошлом, валидация на строго последующем временном окне для исключения Data Leakage).

---

## 3. Метрики качества и Trade-offs

### Контур метрик
* **Основная метрика:** NDCG@5 (оценка точности ранжирования первых 5 позиций).
* **Вспомогательные метрики:** Precision@5, Recall@5, MAP@5.
* **Бизнес-метрика:** Novelty / Diversity (доля категорий, отличных от базовых предпочтений, для избежания «пузыря фильтров»).

### Результаты экспериментов

| Модель / Подход | Precision@5 | Recall@5 | NDCG@5 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Popularity Baseline | 0.1240 | 0.1820 | 0.1450 | ~2 ms |
| CatBoost (Base Features) | 0.2100 | 0.3150 | 0.2240 | ~15 ms |
| CatBoost (Advanced Features) | 0.2450 | 0.3680 | 0.2610 | ~28 ms |

### Архитектурные компромиссы (Trade-offs)
* **Pointwise / Logloss vs Pairwise (YetiRank):** Использован Pointwise-подход на базе `CatBoostClassifier` для быстрой вероятностной оценки $p(click)$. В планах — переход на YetiRank для прямой оптимизации NDCG.
* **Эмбеддинги:** На текущем этапе кандидаты отбираются по бизнес-правилам/свежести. Планируется переход на FAISS (Vector Search) на этапе Candidate Generation.

---

## 4. MLOps: Data Drift & Retraining Pipeline

Для поддержания качества модели в продакшене реализован контур наблюдательности и автоматического обслуживания:

### Детекция дрейфа (`src/drift_detector.py`)
* Отслеживает смещение распределения фичей (`train_ctr`, `history_len`) между `X_train` и реальными входящими логами.
* Использует KS-Test (Kolmogorov-Smirnov) и PSI (Population Stability Index). 
* Сигнал дрейфа генерируется при $PSI > 0.25$.

### Автопереобучение и Quality Gate (`src/retrain.py`)
* Пайплайн запускается по таймеру или при детекции дрейфа.
* Пересчитывает статистику кликов (`item_stats.pkl`) и обучает свежий CatBoost.
* **Quality Gate:** Новая модель перезаписывает рабочие артефакты только при выполнении условия $NDCG@5_{new} > NDCG@5_{old}$ на отложенном тесте.

---

## 5. Структура проекта

```plaintext
pet_project_rec_sys/
├── bot/                    # Код Telegram-бота
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── data/                   # TSV-файлы и обучающие выборки (в .dockerignore)
├── models/                 # Веса модели и артефакты (item_stats.pkl, catboost_ranker.cbm)
├── notebooks/              # Исследовательские Jupyter-ноутбуки (EDA, Validation, Feature Eng.)
├── src/                    # Модули инференса и MLOps
│   ├── app.py              # FastAPI сервис инференса
│   ├── drift_detector.py   # Мониторинг Data Drift (KS-test / PSI)
│   ├── features.py         # Генерация признаков (On-the-fly & Offline)
│   ├── metrics.py          # Расчет NDCG, Precision, Recall
│   ├── retrain.py          # Автоматическое переобучение + Quality Gate
│   └── requirements.txt
├── .dockerignore
├── docker-compose.yml
├── Dockerfile              # Dockerfile для FastAPI API
└── README.md
```