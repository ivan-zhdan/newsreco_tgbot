import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Расчет Population Stability Index (PSI) для непрерывной фичи."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    bins = np.linspace(min(reference.min(), current.min()), max(reference.max(), current.max()), num_bins + 1)
    ref_counts, _ = np.histogram(reference, bins=bins)
    curr_counts, _ = np.histogram(current, bins=bins)

    # Добавляем epsilon для защиты от деления на 0
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts) / len(reference)
    curr_pct = np.where(curr_counts == 0, 1e-4, curr_counts) / len(current)

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi_val)


def check_feature_drift(ref_df: pd.DataFrame, curr_df: pd.DataFrame, numeric_cols: list) -> dict:
    """Проверка дрейфа данных по списку фичей."""
    drift_report = {}
    has_drift = False

    for col in numeric_cols:
        if col not in ref_df.columns or col not in curr_df.columns:
            continue

        # 1. KS-Test (p-value < 0.05 указывает на смещение распределений)
        ks_stat, p_value = ks_2samp(ref_df[col].dropna(), curr_df[col].dropna())

        # 2. PSI calculation
        psi = calculate_psi(ref_df[col].values, curr_df[col].values)

        # Порог PSI: < 0.1 (нет дрейфа), 0.1-0.25 (умеренный), > 0.25 (сильный дрейф)
        is_drifted = p_value < 0.05 and psi > 0.25
        if is_drifted:
            has_drift = True

        drift_report[col] = {
            'p_value': round(p_value, 5),
            'psi': round(psi, 4),
            'drift_detected': is_drifted
        }

    return {'has_drift': has_drift, 'details': drift_report}


if __name__ == "__main__":
    # Загрузка опорного датасета и свежих логов
    ref_data = pd.read_csv('../data/X_train_adv_df.tsv', sep='\t')
    curr_data = pd.read_csv('../data/X_test_adv_df.tsv', sep='\t')

    check_cols = ['train_ctr', 'history_len', 'position_in_impression']
    report = check_feature_drift(ref_data, curr_data, check_cols)

    print(f"Data Drift Detected: {report['has_drift']}")
    for feature, metrics_info in report['details'].items():
        print(f"Feature: {feature} | PSI: {metrics_info['psi']} | Drift: {metrics_info['drift_detected']}")