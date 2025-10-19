import numpy as np
import pandas as pd
from geopy.distance import geodesic

EDGES = [1.00, 7.75, 32.04, 60.84, 94.46, 27390.12]


def create_agg_features(df):
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df.sort_values('transaction_time', inplace=True)
    user_cols = ['name_1', 'name_2', 'one_city', 'us_state']
    agg = (
        df.groupby(user_cols)
            .agg(
            user_trns_count=('amount', 'count'),
            user_amount_mean=('amount', 'mean'),
            user_amount_median=('amount', 'median'),
            user_amount_std=('amount', 'std'),
            user_amount_min=('amount', 'min'),
            user_amount_max=('amount', 'max'),
            user_first_time=('transaction_time', 'min'),
            user_last_time=('transaction_time', 'max')
        )
            .reset_index()
    )

    agg['user_amount_mean'] = np.log1p(agg['user_amount_mean'])
    agg['user_active_days'] = ((agg['user_last_time'] - agg['user_first_time']).dt.days).clip(lower=1)
    agg['user_trns_per_day'] = agg['user_trns_count'] / agg['user_active_days']

    agg.fillna({
        'user_amount_std': 0,
        'user_amount_cv': 0,
        'user_trns_per_day': 0
    }, inplace=True)

    df_merged = df.merge(
        agg[user_cols + ['user_trns_count', 'user_amount_mean', 'user_amount_median', 'user_amount_std',
                         'user_amount_min', 'user_amount_max', 'user_trns_per_day']],
        on=user_cols,
        how='left',
        validate='m:1'
    )

    df.drop(df.index, inplace=True)
    for col in df_merged.columns:
        df[col] = df_merged[col]

    return agg


def apply_agg_features(df, agg):
    user_cols = ['name_1', 'name_2', 'one_city', 'us_state']
    df_merged = df.merge(
        agg[user_cols + ['user_trns_count', 'user_amount_mean', 'user_amount_median', 'user_amount_std',
                         'user_amount_min', 'user_amount_max', 'user_trns_per_day']],
        on=user_cols,
        how='left',
        validate='m:1'
    )

    global_means = agg[
        [
            'user_trns_count',
            'user_amount_mean',
            'user_amount_median',
            'user_amount_std',
            'user_amount_min',
            'user_amount_max',
            'user_trns_per_day'
        ]
    ].mean()

    df_merged.fillna(global_means.to_dict(), inplace=True)
    df.drop(df.index, inplace=True)
    for col in df_merged.columns:
        df[col] = df_merged[col]

    return df


def create_time_features(df):
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['year'] = df['transaction_time'].dt.year
    df['month'] = df['transaction_time'].dt.month
    df['day'] = df['transaction_time'].dt.day
    df['hour'] = df['transaction_time'].dt.hour
    df['minute'] = df['transaction_time'].dt.minute

    df['is_weekend'] = df['transaction_time'].dt.dayofweek >= 5
    df['is_night'] = (df['hour'] >= 0) & (df['hour'] < 6)


def create_distance_features(df):
    df['distance'] = [
        geodesic((lat1, lon1), (lat2, lon2)).km
        for lat1, lon1, lat2, lon2 in zip(
            df['merchant_lat'], df['merchant_lon'],
            df['lat'], df['lon']
        )
    ]


def create_amount_features(df):
    labels = ['1–7.75', '7.75–32.04', '32.04–60.84', '60.84–94.46', '>94.46']
    bins_ext = [-np.inf] + EDGES[1:-1] + [np.inf]
    df['amount_bin5'] = pd.cut(
        df['amount'],
        bins=bins_ext,
        labels=labels,
        include_lowest=True,
        right=True
    )
    df['cents'] = (df['amount'] - np.floor(df['amount'])).clip(lower=0)
    df['amount'] = np.log1p(df['amount'].clip(lower=0))
    df['cents'] = np.log1p(df['cents'].clip(lower=0))


def create_merch_jobs_features(df):
    df['merch_len'] = df['merch'].str.len()
    df['jobs_len'] = df['jobs'].str.len()
    df['merch_word_cnt'] = df['merch'].str.split().str.len()
    df['population_city'] = np.log1p(df['population_city'].clip(lower=0))


def create_population_features(df):
    df['population_city'] = np.log1p(df['population_city'].clip(lower=0))


def create_post_code_features(df):
    df['post_code'] = df['post_code'].astype(str)
    df['zip1'] = df['post_code'].str[:1]
    df['zip3'] = df['post_code'].str[1:3]


def fit_city_stats(df):
    counts = df['one_city'].astype(str).value_counts(dropna=False)
    stats = counts.rename('city_count').reset_index().rename(columns={'index': 'one_city'})
    return stats


def apply_city_features(df, stats):
    freq = df['one_city'].map(dict(zip(stats['one_city'], stats['city_count']))).fillna(0).astype(int)
    df['city_freq'] = freq
    df['city_freq_log'] = np.log1p(df['city_freq'])
    df['city_rareness'] = 1 / (df['city_freq_log'] + 1)
