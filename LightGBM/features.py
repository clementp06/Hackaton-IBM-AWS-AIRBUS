import numpy as np
import pandas as pd

from data import ID_COLUMNS


ROLLING_WINDOWS = [3, 6, 12, 24]


def add_dates(environment):
    environment = environment.copy()
    environment["date"] = pd.to_datetime(environment["year_month"] + "-01")
    return environment.sort_values(["aircraft_id", "date"]).reset_index(drop=True)


def get_numeric_environment_columns(environment):
    return environment.select_dtypes(include="number").columns.tolist()


def make_calendar_features(environment):
    month = environment["date"].dt.month
    return pd.DataFrame(
        {
            "calendar_year": environment["date"].dt.year,
            "calendar_month": month,
            "calendar_month_sin": np.sin(2 * np.pi * month / 12),
            "calendar_month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=environment.index,
    )


def make_cumulative_features(environment, groups, numeric_columns, history_count):
    cumulative_sum = groups[numeric_columns].cumsum()
    cumulative_mean = cumulative_sum.div(history_count, axis=0)
    cumulative_std = (
        groups[numeric_columns]
        .expanding(min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

    return [
        cumulative_sum.add_suffix("__history_sum"),
        cumulative_mean.add_suffix("__history_mean"),
        groups[numeric_columns].cummin().add_suffix("__history_min"),
        groups[numeric_columns].cummax().add_suffix("__history_max"),
        cumulative_std.add_suffix("__history_std"),
        (environment[numeric_columns] - cumulative_mean).add_suffix(
            "__diff_from_history_mean"
        ),
    ]


def make_rolling_features(environment, groups, numeric_columns):
    feature_parts = []

    for window in ROLLING_WINDOWS:
        rolling = groups[numeric_columns].rolling(window, min_periods=1)
        rolling_mean = rolling.mean().reset_index(level=0, drop=True)
        rolling_std = rolling.std().reset_index(level=0, drop=True)

        feature_parts.extend(
            [
                rolling_mean.add_suffix(f"__last_{window}_mean"),
                rolling_std.add_suffix(f"__last_{window}_std"),
                (environment[numeric_columns] - rolling_mean).add_suffix(
                    f"__diff_from_last_{window}_mean"
                ),
            ]
        )

    return feature_parts


def make_lag_features(environment, groups, numeric_columns):
    lag1 = groups[numeric_columns].shift(1)
    lag3 = groups[numeric_columns].shift(3)

    return [
        lag1.add_suffix("__lag1"),
        lag3.add_suffix("__lag3"),
        (environment[numeric_columns] - lag1).add_suffix("__delta_lag1"),
        (environment[numeric_columns] - lag3).add_suffix("__delta_lag3"),
    ]


def build_history_feature_table(environment):
    environment = add_dates(environment)
    numeric_columns = get_numeric_environment_columns(environment)
    groups = environment.groupby("aircraft_id", sort=False)
    history_count = groups.cumcount() + 1

    feature_parts = [
        environment[numeric_columns].add_prefix("current__"),
        pd.DataFrame({"history_count": history_count}, index=environment.index),
        make_calendar_features(environment),
    ]
    feature_parts.extend(
        make_cumulative_features(environment, groups, numeric_columns, history_count)
    )
    feature_parts.extend(make_rolling_features(environment, groups, numeric_columns))
    feature_parts.extend(make_lag_features(environment, groups, numeric_columns))

    features = pd.concat(feature_parts, axis=1).astype("float32")
    keys = environment[ID_COLUMNS].copy()
    return pd.concat([keys, features], axis=1)
