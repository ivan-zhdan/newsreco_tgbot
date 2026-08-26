import features
import metrics
from catboost import CatBoostClassifier, Pool
import joblib

# Загрузка и разделение данных
news_df, behav_df = features.load_data('../data/news.tsv', '../data/behaviors.tsv')
train_df, test_df, val_df = features.behav_df_time_split(behav_df)

# Рассчитываем статистику
item_clicks, item_impressions = features.extract_item_stats(train_df)

# Генерируем фичи
X_train_df = features.prepare_advanced_dataset(
    train_df, news_df, item_clicks, item_impressions, is_train=True
)
X_val_df = features.prepare_advanced_dataset(
    val_df, news_df, item_clicks, item_impressions, is_train=True # is_train=True для наличия таргета
)

# Убираем Nan
X_train_df.dropna(subset=['target'], inplace=True)
X_train_df['target'] = X_train_df['target'].astype(int)
X_val_df.dropna(subset=['target'], inplace=True)
X_val_df['target'] = X_val_df['target'].astype(int)

# Подготовка фичей для модели
feature_cols = [
    'position_in_impression', 'history_len', 'train_clicks', 'train_impressions', 'train_ctr',
    'is_fav_category_match', 'category_preference_score', 'item_category', 'user_fav_category'
]
cat_features = ['item_category', 'user_fav_category']

X_train, y_train = X_train_df[feature_cols], X_train_df['target']
X_val, y_val = X_val_df[feature_cols], X_val_df['target']

# Балансировка классов
scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

# Обучение
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.01,
    #scale_pos_weight=scale_weight,
    loss_function='Logloss',
    random_seed=42,
    verbose=100
)
model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

# Валидация
X_val_df['score'] = model.predict_proba(X_val)[:, 1]
val_ndcg = metrics.evaluate_ranking(X_val_df, metric_fn=metrics.ndcg_at_k, k=5)
val_recall = metrics.evaluate_ranking(X_val_df, metric_fn=metrics.recall_at_k, k=5)

print(f"Validation NDCG@5: {val_ndcg:.4f} | Recall@5: {val_recall:.4f}")
# Сохраняем модель
model.save_model('../models/catboost_ranker.cbm')
joblib.dump({'clicks': item_clicks, 'impressions': item_impressions}, '../models/item_stats.pkl')