import catboost
import time
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


DATASET_FILENAME="multilabel_classification_v3_50000.csv"


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


def visualize(data: pd.DataFrame) -> None:
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 10), dpi=80)

    label_font = {
        'family': 'serif',
        'color': 'darkblue',
        'size': 10
    }

    plt.xlabel('Feature 1', fontdict=label_font)
    plt.ylabel('Feature 2', fontdict=label_font)
    plt.title('Multilabel classification')

    color_map = {
        0: 'red',
        1: 'yellow',
        2: 'blue'
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
    print(f"\tCatBoost : {catboost.__version__}")
    print(f"\tPandas   : {pd.__version__}")
    print(f"\tNumPy    : {np.__version__}")
    print("\n")


    # Prepare data
    if not os.path.exists(DATASET_FILENAME):
        print(f"Error : dataset <{DATASET_FILENAME}> not found")
        quit(1)

    df_multilabel_classification = pd.read_csv(DATASET_FILENAME)
    show_info(df_multilabel_classification)

    target = "class_label"
    examples_count = df_multilabel_classification.shape[0]
    class_labels = df_multilabel_classification[target]

    rows_list = []
    for i in range(examples_count):
        row = {
            "A": 0,
            "B": 0,
            "C": 0
        }

        if class_labels[i] == 0:
            row["A"] = 1

        if class_labels[i] == 1:
            row["B"] = 1

        if class_labels[i] == 2:
            row["C"] = 1

        rows_list.append(row)

    Y = pd.DataFrame(rows_list)
    X = df_multilabel_classification.drop(target, axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    train_pool, test_pool = Pool(X_train, Y_train), Pool(X_test, Y_test)


    # Create and fit model
    model = CatBoostClassifier(
        loss_function='MultiLogloss',
        class_names=['A', 'B', 'C'],
        iterations=2000,
        depth=5,
        learning_rate=0.05,
        metric_period=10,
        random_seed=789
    )

    print("Start model training")
    start_train_timestamp = time.time()
    model.fit(
        X_train,
        Y_train,
        metric_period=5,
        eval_set=(X_test, Y_test)
    )
    end_of_train_timestamp = time.time()
    train_time_elapsed = end_of_train_timestamp - start_train_timestamp
    print(f"Time spent training the model : {train_time_elapsed:.2f} s")


    # Get predictions and calculate metrics
    predicts = model.predict(X_test)
    predicts_count = len(predicts)
    predicted_labels = []
    for i in range(predicts_count):
        label = 0

        if predicts[i][1] == 1:
            label = 1

        if predicts[i][2] == 1:
            label = 2

        predicted_labels.append(label)


    #rmse_score = root_mean_squared_error(Y_test, Y_predicted)
    #mae_score = mean_absolute_error(Y_test, Y_predicted)


    # Visualize
    results = X_test.copy()
    results.reset_index(drop=True, inplace=True)
    results["class_label"] = predicted_labels
    visualize(results)


    print("Success")