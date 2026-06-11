import pandas as pd


TARGET_COLUMN = "corrosion_risk"
ID_COLUMNS = ["aircraft_id", "year_month"]


def split_prediction_id(labels):
    labels = labels.copy()
    labels[ID_COLUMNS] = labels["id"].str.rsplit("_", n=1, expand=True)
    return labels


def load_environment(path):
    return pd.read_csv(path)


def load_labels(path):
    labels = split_prediction_id(pd.read_csv(path))
    labels[TARGET_COLUMN] = labels[TARGET_COLUMN].astype(int)
    return labels[ID_COLUMNS + [TARGET_COLUMN]]


def build_training_table(environment, labels, feature_table):
    training = labels.merge(
        feature_table,
        on=ID_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    missing_rows = len(labels) - len(training)
    if missing_rows:
        print(f"Attention: {missing_rows} labels sans features environnement.")

    feature_columns = [
        column
        for column in training.columns
        if column not in ID_COLUMNS + [TARGET_COLUMN]
    ]
    X = training[feature_columns]
    y = training[TARGET_COLUMN]
    groups = training["aircraft_id"]
    return X, y, groups, feature_columns
