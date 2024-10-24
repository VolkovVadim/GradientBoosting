import catboost
import os

import pandas as pd

from catboost import cv
from catboost import Pool


DATASET_FILENAME="regression_v3_10000.csv"
TREES_COUNT = 10000
FOLDS_COUNT = 5


def show_info(dataframe: pd.DataFrame) -> None:
    print(f"Dataframe :\n{dataframe.head(30)}\n")
    print(f"Records count    : {dataframe.shape[0]}\n")
    print(f"NaN values count :\n{dataframe.isna().sum()}")

    column_names = list(dataframe)
    max_len = len(max(column_names, key=len))
    if max_len < 10:
        max_len = 10

    print(f"Data types :")
    for column_name, data_type in dataframe.dtypes.items():
        print(f"  {column_name}{' ' * (max_len - len(column_name))} : <{data_type}>")

    print("\n")


def print_cv_summary(cv_data: pd.DataFrame) -> None:
    best_value = cv_data['test-RMSE-mean'].min()
    best_iter = cv_data['iterations'].values[cv_data['test-RMSE-mean'].idxmin()]
    std_dev = cv_data['test-RMSE-std'].values[cv_data['test-RMSE-mean'].idxmin()]

    print('Best validation RMSE score : {:.4f}±{:.4f} on step {}'.format(
        best_value,
        std_dev,
        best_iter)
    )


if __name__ == "__main__":
    print("Versions")
    print(f"\tCatBoost : {catboost.__version__}")
    print(f"\tPandas   : {pd.__version__}")

    # Prepare data
    if not os.path.exists(DATASET_FILENAME):
        print(f"Error : dataset <{DATASET_FILENAME}> not found")
        quit(1)

    df_regression = pd.read_csv(DATASET_FILENAME)
    show_info(df_regression)

    target = "value"
    Y = df_regression[target]
    X = df_regression.drop(target, axis=1)

    train_pool = Pool(data=X, label=Y)

    # parameters for training model inside cross validation:
    cv_params = {
        "objective": "RMSE",
        "depth": 5,
        "iterations": TREES_COUNT,
        "learning_rate": 0.025
    }

    print(f"Trees count : {TREES_COUNT}, Folds count : {FOLDS_COUNT}")
    input("Press any key to continue...")

    # Cross validation
    cv_data = cv(
        params=cv_params,
        dtrain=train_pool,
        fold_count=FOLDS_COUNT,
        shuffle=True,
        #stratified=True,
        metric_period=500,
        verbose=True
    )

    show_info(cv_data)
    print_cv_summary(cv_data)

    print("Success")