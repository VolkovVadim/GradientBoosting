import os
import time

import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


DATASET_FILENAME="regression_v3_50000_with_noise.csv"


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
    print(f"XGBoost version : {xgb.__version__}")

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
    regressor = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=5,
        learning_rate=0.05,
        n_estimators=1000
    )

    print("Start model training")
    start_train_timestamp = time.time()
    regressor.fit(train_X, train_Y)
    end_of_train_timestamp = time.time()
    train_time_elapsed = end_of_train_timestamp - start_train_timestamp
    print(f"Time spent training the model : {train_time_elapsed:.2f} s")


    # Get predictions
    pred_Y       = regressor.predict(test_X)


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
