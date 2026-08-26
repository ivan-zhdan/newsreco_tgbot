import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
import features
import metrics

MODEL_PATH = '../models/catboost_ranker.cbm'
STATS_PATH = '../models/item_stats.pkl'
MIN_NDCG_THRESHOLD = 0.20  # Минимально допустимый порог качества


def run_retraining_pipeline():
    print("--- [1/5] Загрузка свежих данных ---")
    news_df, behav_df = features.load_data('../data/news.tsv', '../data/behaviors.tsv')
    train_df, val_df, _ = features.behav_df_time_split(behav_df)

    print("--- [2/5] Извлечение глобальной статистики популярности ---")
    item_clicks, item_impressions = features.extract_item_stats(train_df)

    print("--- [3/5] Генерация продвинутого набора признаков ---")
    X_train_df = features.prepare_advanced_dataset(
        train_df, news_df, item_clicks, item_impressions, is_train=True
    )
    X_val_df = features.prepare_advanced_dataset(
        val_df, news_df, item_clicks, item_impressions, is_train=True
    )

    # Очистка пропущенных меток
    X_train_df.dropna(subset=['target'], inplace=True)
    X_train_df['target'] = X_train_df['target'].astype(int)
    X_val_df.dropna(subset=['target'], inplace=True)
    X_val_df['target'] = X_val_df['target'].astype(int)

    feature_cols = [
        'position_in_impression', 'history_len', 'train_clicks', 'train_impressions', 'train_ctr',
        'is_fav_category_match', 'category_preference_score', 'item_category', 'user_fav_category'
    ]
    cat_features = ['item_category', 'user_fav_category']

    X_train, y_train = X_train_df[feature_cols], X_train_df['target']
    X_val, y_val = X_val_df[feature_cols], X_val_df['target']

    scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    print("--- [4/5] Обучение новой версии CatBoost ---")
    new_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        loss_function='Logloss',
        random_seed=42,
        verbose=0
    )
    new_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

    # Оценка качества
    X_val_df['score'] = new_model.predict_proba(X_val)[:, 1]
    new_ndcg = metrics.evaluate_ranking(X_val_df, metric_fn=metrics.ndcg_at_k, k=5)
    print(f"Новая модель validation NDCG@5: {new_ndcg:.4f}")

    print("--- [5/5] Quality Gate & Деплой артефактов ---")
    # Проверяем, превосходит ли новая модель минимальный порог
    if new_ndcg >= MIN_NDCG_THRESHOLD:
        print("Quality Gate пройден. Обновление файлов моделей...")

        # Атомарная запись артефактов
        new_model.save_model(MODEL_PATH)
        joblib.dump({'clicks': item_clicks, 'impressions': item_impressions}, STATS_PATH)
        print("Успешно: Новая модель и артефакты сохранены на диск!")
    else:
        print(f"Quality Gate НЕ пройден! (NDCG@5: {new_ndcg:.4f} < {MIN_NDCG_THRESHOLD}). "
              "Обновление отменено, сохранена старая модель.")


if __name__ == "__main__":
    run_retraining_pipeline()