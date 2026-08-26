import pandas as pd
import numpy as np
from collections import Counter
import tqdm

def load_data(news_path, behaviors_path):

    '''reads datasets, makes nessesary additions'''

    news_df=pd.read_csv(news_path, sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])


    behav_df=pd.read_csv(behaviors_path, sep='\t', names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    behav_df['datetime'] = pd.to_datetime(behav_df['time'], format='%m/%d/%Y %I:%M:%S %p')
    behav_df = behav_df[behav_df['time'] != 'time'].copy()

    return news_df, behav_df


def behav_df_time_split(behav_df):

    '''splits behav dataframe into train and test'''

    train_df = behav_df[behav_df['datetime'] < '14-11-2019']
    test_and_val_df = behav_df[behav_df['datetime'] >= '14-11-2019']
    test_df=test_and_val_df.sample(frac=0.5, random_state=42)
    val_df=test_and_val_df.drop(test_df.index)

    return train_df, test_df, val_df


def extract_item_stats(df):
    item_clicks = Counter()
    item_impressions = Counter()

    for imp in df['impressions'].dropna():
        items = str(imp).split()
        for item in items:
            parts = item.rsplit('-', 1)  # rsplit с конца на 2 части на случай, если в news_id есть '-'
            if len(parts) == 2:
                news_id, label_str = parts[0], parts[1]
                try:
                    label = int(label_str)
                    item_impressions[news_id] += 1
                    if label == 1:
                        item_clicks[news_id] += 1
                except ValueError:
                    # Если за дефисом шло не число
                    continue
            else:
                # В строке нет дефиса с меткой
                continue

    return item_clicks, item_impressions


def prepare_advanced_dataset(df, news_category_dict, train_clicks, train_impressions, is_train=True):


    rows = []

    # Для прогресс-бара
    df_iter = tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing sessions")

    for idx, row in df_iter:
        imp_str = row['impressions']
        if pd.isna(imp_str): continue

        session_time = pd.to_datetime(row['time'], format='%m/%d/%Y %I:%M:%S %p')

        # --- ЮЗЕР ФИЧИ (ПЕРСОНАЛИЗАЦИЯ) ---
        history = str(row['history']).split() if pd.notna(row['history']) else []
        history_len = len(history)

        # Получаем категории из истории
        hist_categories = [news_category_dict.get(nid) for nid in history if news_category_dict.get(nid)]
        hist_cat_counts = Counter(hist_categories)
        total_hist_cats = len(hist_categories)

        # Топ-1 категория пользователя
        fav_cat = hist_cat_counts.most_common(1)[0][0] if hist_cat_counts else None

        items = imp_str.split()
        for pos, item in enumerate(items):
            parts = item.split('-')
            news_id = parts[0]
            # В train используем метки, в test - только кандидатов (pointwise approach)
            label = int(parts[1]) if is_train or '-' in item else None

            # --- ИТЕМ ФИЧИ ---
            item_cat = news_category_dict.get(news_id)

            clicks = train_clicks.get(news_id, 0)
            imps = train_impressions.get(news_id, 0)
            ctr = clicks / imps if imps > 0 else 0.0

            # --- ВЗАИМОДЕЙСТВИЕ (ADVANCED) ---
            # 1. Категориальное совпадение
            is_fav_cat = (item_cat == fav_cat) if fav_cat and item_cat else False

            # 2. Доля этой категории в истории
            cat_pref_score = (hist_cat_counts.get(item_cat, 0) / total_hist_cats) if total_hist_cats > 0 and item_cat else 0.0

            rows.append({
                'session_id': idx,
                'news_id': news_id,
                # Старые фичи
                'position_in_impression': pos,
                'history_len': history_len,
                'train_clicks': clicks,
                'train_impressions': imps,
                'train_ctr': ctr,
                # Категориальные (нужно указать CatBoost, что это категории)
                'item_category': item_cat if item_cat else "UNKNOWN",
                'user_fav_category': fav_cat if fav_cat else "UNKNOWN",
                # Новые advanced фичи
                'is_fav_category_match': int(is_fav_cat),
                'category_preference_score': cat_pref_score,
                # Таргет
                'target': label
            })

    return pd.DataFrame(rows)