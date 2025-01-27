import catboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier


MODEL_FILENAME = "binary_classification_v3_n200.cbm"
INPUT_DATASET = "binary_classification_v3_5000_no_label.csv"


def show_model_info(model) -> None:
    model_params = model.get_all_params()

    selected_params = [
        "iterations",
        "depth",
        "grow_policy",
        "max_leaves"
    ]

    print("Model info :")
    for param_name in selected_params:
        print(f"    {param_name}{' ' * (20 - len(param_name))} : {model_params[param_name]}")

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
    plt.title(f'Binary classificator predictions ({points_count} points)')

    color = ['red' if label == 0 else 'blue' for label in data.class_label]

    point_size = 5 if data.shape[0] < 50000 else 1

    plt.scatter(
        data.feature_1,
        data.feature_2,
        s=point_size,
        c=color
    )

    plt.show()


if __name__ == "__main__":
    print(f"CatBoost version : {catboost.__version__}")

    classifier = CatBoostClassifier()
    classifier.load_model(MODEL_FILENAME)
    show_model_info(classifier)

    print(f"Loading data from <{INPUT_DATASET}>...")
    df_input = pd.read_csv(INPUT_DATASET)
    print("Loaded\n")

    print("Get model predictions...")
    predictions = classifier.predict(df_input)

    df_data = df_input.copy()
    df_data["class_label"] = predictions
    visualize(df_data)

    print("Done")
