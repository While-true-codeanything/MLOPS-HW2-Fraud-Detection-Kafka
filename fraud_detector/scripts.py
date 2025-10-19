import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing import apply_agg_features, create_time_features, create_distance_features, \
    create_amount_features, create_merch_jobs_features, create_population_features, create_post_code_features, \
    apply_city_features

THRESHOLD = 0.379


def preprocess_df(df_test, user_stats, city_stats):
    apply_agg_features(df_test, user_stats)
    create_time_features(df_test)
    create_distance_features(df_test)
    create_amount_features(df_test)
    create_merch_jobs_features(df_test)
    create_population_features(df_test)
    create_post_code_features(df_test)
    apply_city_features(df_test, city_stats)
    df_test.drop(columns=['transaction_time', 'merchant_lat', 'merchant_lon', 'lat', 'lon', 'post_code', 'street'],
                 inplace=True)
    return df_test


def get_pred_probs(model, df_test):
    return model.predict_proba(df_test)[:, 1]


def get_predictions(y_test_proba, path):
    df_test_predict = pd.DataFrame(y_test_proba, columns=["prediction"]).reset_index()
    df_test_predict["prediction"] = df_test_predict["prediction"].apply(
        lambda x: 1 if x >= THRESHOLD else 0
    )
    df_test_predict.to_csv(f'predict_{path.name}', index=False)


def get_topn_feature_importance(model, df_test, output, n=5):
    model_feature_importance = pd.DataFrame(
        data={
            "features": df_test.columns.to_list(),
            "score": model.get_feature_importance(),
        }
    )
    model_feature_importance = model_feature_importance.sort_values(by="score", ascending=False).head(n)
    top_dict = dict(zip(model_feature_importance['features'], model_feature_importance['score'].astype(float)))
    with open(output + "//feature_importance.json", 'w', encoding='utf-8') as f:
        json.dump(top_dict, f, ensure_ascii=False, indent=2)


def get_predict_density_distribution(y_test_proba, output):
    plt.figure(figsize=(12, 8))
    plt.hist(np.asarray(y_test_proba), bins=100, density=True, alpha=0.8)
    plt.grid(alpha=0.5)
    plt.tight_layout()

    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.title("Distribution of predicted density")
    plt.savefig(output + "//density_distribution.png")
    plt.close()
