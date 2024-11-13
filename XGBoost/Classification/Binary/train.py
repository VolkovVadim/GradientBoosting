import os
import time

import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


DATASET_FILENAME="binary_classification_v3_50000_with_noise.csv"


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
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 10), dpi=80)

    label_font = {
        'family': 'serif',
        'color': 'darkblue',
        'size': 10
    }

    points_count = data.shape[0]

    plt.xlabel('Feature 1', fontdict=label_font)
    plt.ylabel('Feature 2', fontdict=label_font)
    plt.title(f"Binary classification ({points_count} points)")

    color_map = {
        0: 'red',
        1: 'blue',
        2: 'green'
    }

    color = [color_map[label] for label in data.class_label]
    point_size = 5 if data.shape[0] < 50000 else 1

    plt.scatter(
        data.feature_1,
        data.feature_2,
        s=point_size,
        c=color
    )

    plt.show()


if __name__ == "__main__":
    print("Versions")
    print(f"  XGBoost : {xgb.__version__}")
    print(f"  Pandas  : {pd.__version__}")


    # Load data
    if not os.path.exists(DATASET_FILENAME):
        print(f"Error : dataset <{DATASET_FILENAME}> not found")
        quit(1)

    df_binary_classification = pd.read_csv(DATASET_FILENAME)
    show_info(df_binary_classification)


    # Prepare data
    train_df, test_df = train_test_split(df_binary_classification, test_size=0.2, random_state=789)
    print(f"\nexamples count : train_df : <{train_df.shape[0]}>, test_df : <{test_df.shape[0]}>")

    target = "class_label"
    train_Y, train_X = train_df[target], train_df.drop(target, axis=1)
    test_Y,  test_X  = test_df[target],  test_df.drop(target, axis=1)

    print(f"Labels : {set(df_binary_classification[target])}\n")


    # Create and fit model
    classificator = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=5,
        learning_rate=0.01,
        n_estimators=2500,
        random_state=789
    )

    print("Start model training")
    start_train_timestamp = time.time()
    classificator.fit(train_X, train_Y)
    end_of_train_timestamp = time.time()
    train_time_elapsed = end_of_train_timestamp - start_train_timestamp
    print(f"Time spent training the model : {train_time_elapsed:.2f} s")


    # Get predictions
    predicted_Y = classificator.predict(test_X)


    # Visualize
    results = test_X.assign(class_label=predicted_Y)
    visualize(results)


    print("Success")
