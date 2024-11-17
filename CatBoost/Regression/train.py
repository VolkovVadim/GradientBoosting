import catboost
import time
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


DATASET_FILENAME="regression_v3_10000.csv"


def show_info(dataframe: pd.DataFrame) -> None:
    print(f"Dataframe :\n{dataframe.head(20)}\n")
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


def visualize(data: pd.DataFrame) -> None:
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(12, 10), dpi=80)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("feature_1")
    ax.set_ylabel("feature_2")
    ax.set_zlabel("value")

    ax.xaxis.label.set_color("blue")
    ax.yaxis.label.set_color("blue")
    ax.zaxis.label.set_color("red")

    color = data.value
    points_count = data.shape[0]
    point_size = 5 if points_count < 50000 else 1

    ax.text2D(0.05, 0.95, f"Model predictions ({points_count} examples)", transform=ax.transAxes)

    ax.scatter(
        data['feature_1'],
        data['feature_2'],
        data['value'],
        c = color,           # values for cmap
        s = point_size,      # marker size
        cmap='viridis'
    )

    plt.show()


if __name__ == "__main__":
    print("Versions")
    print(f"\tCatBoost : {catboost.__version__}")
    print(f"\tPandas   : {pd.__version__}")
    print(f"\tNumPy    : {np.__version__}")
    print("\n")


    # Prepare data
    if not os.path.exists(DATASET_FILENAME):
        print(f"Error : dataset <{DATASET_FILENAME}> not found")
        quit(1)

    df_regression = pd.read_csv(DATASET_FILENAME)
    show_info(df_regression)

    train_df, test_df = train_test_split(df_regression, test_size=0.2, random_state=789)
    print(f"\nexamples count : train_df : <{train_df.shape[0]}>, test_df : <{test_df.shape[0]}>")

    target = "value"
    train_Y, train_X = train_df[target], train_df.drop(target, axis=1)
    test_Y,  test_X  = test_df[target],  test_df.drop(target, axis=1)


    # Create and fit model
    model = CatBoostRegressor(
        objective='RMSE',
        depth=5,
        iterations=10000,
        learning_rate=0.025
    )

    print("Start model training")
    start_train_timestamp = time.time()
    model.fit(
        train_X,
        train_Y,
        metric_period=250
        #plot=True
    )
    end_of_train_timestamp = time.time()
    train_time_elapsed = end_of_train_timestamp - start_train_timestamp
    print(f"Time spent training the model : {train_time_elapsed:.2f} s")


    # Get predictions
    pred_Y       = model.predict(test_X)


    # Calculate metrics
    rmse_score = root_mean_squared_error(test_Y, pred_Y)
    mae_score = mean_absolute_error(test_Y, pred_Y)

    print(f"RMSE : {rmse_score:.2f}")
    print(f"MAE  : {mae_score:.2f}")


    # Visualize results
    results = test_X.copy()
    results.reset_index(drop=True, inplace=True)
    results["value"] = pred_Y
    visualize(results)


    print("Success")