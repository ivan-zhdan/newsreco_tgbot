import numpy as np

def recall_at_k(actual, predicted, k=5):
    predicted_at_k = predicted[:k]
    hits = len(set(actual) & set(predicted_at_k))
    return hits / len(actual) if len(actual) > 0 else 0.0

def ndcg_at_k(actual, predicted, k=5):
    predicted_at_k = predicted[:k]
    dcg = 0.0
    for i, p in enumerate(predicted_at_k):
        if p in actual:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual), k))])
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(df, metric_fn, k=5, score_col='score'):
    """
    Группирует датафрейм по session_id и рассчитывает усредненную метрику ранжирования.
    """
    metric_values = []

    # Группируем по сессиям
    sessions = df.groupby('session_id')

    for _, session_data in sessions:
        # Извлекаем реальные клики пользователя
        actual_clicks = session_data[session_data['target'] == 1]['news_id'].tolist()
        if not actual_clicks:
            continue

        # Ранжируем рекомендации по вероятностям модели (score)
        ranked_recommendations = session_data.sort_values(by=score_col, ascending=False)['news_id'].tolist()

        # Считаем метрику для текущей сессии
        metric_values.append(metric_fn(actual_clicks, ranked_recommendations, k=k))

    return float(np.mean(metric_values)) if metric_values else 0.0