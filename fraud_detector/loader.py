import pandas as pd
from catboost import CatBoostClassifier


#def load_test(path):
# df_test = pd.read_csv(f"{path}//test.csv")
#   return df_test


def load_model(path):
    model = CatBoostClassifier()
    model.load_model(f"{path}//catboost_model.cbm")
    return model


def load_stats(path):
    user_stats = pd.read_csv(f"{path}//user_stats.csv")
    city_stats = pd.read_csv(f"{path}//city_stats.csv")
    return user_stats, city_stats
