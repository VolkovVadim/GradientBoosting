import catboost
import time
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, confusion_matrix, accuracy_score, roc_auc_score
from matplotlib.pyplot import figure


DATASET_FILENAME="binary_classification_v3_100000_with_noise.csv"
BORDERS_FILENAME="binary_classification_v3_border.csv"


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


def visualize(predictions: pd.DataFrame, confusion: np.ndarray, borders: pd.DataFrame = None) -> None:
    plt.style.use('ggplot')

    label_font = {
        'family': 'serif',
        'color': 'darkblue',
        'size': 10
    }

    points_count = predictions.shape[0]

    # Initialise the subplot function using number of rows and columns
    _, axis = plt.subplots(ncols=2, figsize=(18, 8), dpi=80, num='Results')

    predictions_axis, heatmap_axis = axis[0], axis[1]


    predictions_axis.set_xlabel('Feature 1', fontdict=label_font, labelpad=5)
    predictions_axis.set_ylabel('Feature 2', fontdict=label_font, labelpad=5)
    predictions_axis.set_title(f'Binary classification ({points_count} predictions)')

    if borders is not None:
        predictions_axis.plot(
            borders.x1,
            borders.y1,
            linestyle='--',
            linewidth=2,
            #c='midnightblue',
            #c='indigo'
            c='black'
        )

    color_map = {
        0: 'red',
        1: 'blue',
        2: 'green'
    }

    color = [color_map[label] for label in predictions.class_label]
    point_size = 5 if predictions.shape[0] < 50000 else 1

    predictions_axis.scatter(
        predictions.feature_1,
        predictions.feature_2,
        s=point_size,
        c=color
    )

    heatmap_axis.set_title("Confusion matrix")

    sns.heatmap(
        confusion,
        annot=True,
        fmt='d',
        cmap='summer',
        xticklabels=['Predicted False', 'Predicted True'],
        yticklabels=['Actual False', 'Actual True'],
        ax=heatmap_axis
    )

    heatmap_axis.set_xlabel('Predicted', fontdict=label_font, labelpad=5)
    heatmap_axis.set_ylabel('Actual', fontdict=label_font, labelpad=5)

    plt.show()


def visualize_roc(predictions_proba, actual_labels):
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 10), dpi=80)

    FPR, TPR, _ = metrics.roc_curve(actual_labels, predictions_proba)
    AUC = metrics.roc_auc_score(actual_labels, predictions_proba)

    plt.title(f"AUC ROC : {AUC:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(FPR, TPR)

    plt.show()


if __name__ == "__main__":
    print("Versions")
    print(f"\tCatBoost : {catboost.__version__}")
    print(f"\tPandas   : {pd.__version__}")
    print(f"\tNumPy    : {np.__version__}")
    print("\n")


    # Load data
    if not os.path.exists(DATASET_FILENAME):
        print(f"Error : dataset <{DATASET_FILENAME}> not found")
        quit(1)

    df_regression = pd.read_csv(DATASET_FILENAME)
    show_info(df_regression)


    # Prepare data
    train_df, test_df = train_test_split(df_regression, test_size=0.2, random_state=789)
    print(f"\nexamples count : train_df : <{train_df.shape[0]}>, test_df : <{test_df.shape[0]}>")

    target = "class_label"
    train_Y, train_X = train_df[target], train_df.drop(target, axis=1)
    test_Y,  test_X  = test_df[target],  test_df.drop(target, axis=1)

    print(f"Labels : {set(df_regression[target])}\n")


    # Create and fit model
    TREES_COUNT = 200

    model = CatBoostClassifier(
        iterations=TREES_COUNT,
        depth=5,
        learning_rate=0.05,
        random_seed=789
    )

    print("Start model training")
    start_train_timestamp = time.time()
    model.fit(
        train_X,
        train_Y,
        metric_period=5,
        eval_set=(test_X, test_Y)
    )
    end_of_train_timestamp = time.time()
    train_time_elapsed = end_of_train_timestamp - start_train_timestamp
    print(f"Time spent training the model : {train_time_elapsed:.2f} s")


    # Get predictions
    pred_Y = model.predict(test_X)
    pred_Y_proba = model.predict_proba(test_X)[::,1]


    # Calculate metrics
    test_values = list(test_Y)
    correct, total = 0, len(test_values)
    for index in range(total):
        if test_values[index] == pred_Y[index]:
            correct += 1

    incorrect = total - correct

    rmse_score = root_mean_squared_error(test_Y, pred_Y)
    mae_score = mean_absolute_error(test_Y, pred_Y)
    accuracy_v1 = accuracy_score(test_Y, pred_Y)
    confusion = confusion_matrix(test_Y, pred_Y)
    auc_roc = roc_auc_score(test_Y, pred_Y)

    TN, FP, FN, TP = confusion.ravel()
    accuracy_v2 = (TP + TN) / (TP + TN + FP + FN)
    precision   = TP / (TP + FP)
    recall      = TP / (TP + FN)
    f1_measure  = TP / (TP + ((FP + FN) / 2))

    print(f"Results (correct : {correct}, incorrect : {incorrect}, accuracy : {correct / total})")
    print(f"  RMSE         : {rmse_score:.2f}")
    print(f"  MAE          : {mae_score:.2f}")
    print(f"  Accuracy (1) : {accuracy_v1:.3f}")
    print(f"  Accuracy (2) : {accuracy_v2:.3f}")
    print(f"  Precision    : {precision:.2f}")
    print(f"  Recall       : {recall:.2f}")
    print(f"  F1 measure   : {f1_measure:.2f}")
    print(f"  AUC ROC      : {auc_roc:.2f}")


    # Visualize
    df_borders = None
    if os.path.exists(BORDERS_FILENAME):
        df_borders = pd.read_csv(BORDERS_FILENAME)

    results = test_X.assign(class_label=pred_Y)
    visualize(predictions=results, confusion=confusion, borders=df_borders)
    #visualize_roc(predictions_proba=pred_Y_proba, actual_labels=test_Y)


    # Save model to file
    model_filename = f"binary_classification_n{TREES_COUNT}.cbm"  # extension is optional
    model.save_model(model_filename, format="cbm")
    print(f"Model saved to {model_filename}")


    print("Success")