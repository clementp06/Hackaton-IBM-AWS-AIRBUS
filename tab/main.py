import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

env_training = pd.read_csv("../data/environment_training.csv")
corr_training = pd.read_csv("../data/corrosions_training.csv")



dates = corr_training["observation_date"]


corr_training["observation_date"] = pd.to_datetime(corr_training["observation_date"])
corr_training = corr_training.assign(
    months=(
        (corr_training["observation_date"].dt.year - corr_training["aircraft_delivery_year"]) * 12
        + (corr_training["observation_date"].dt.month - corr_training["aircraft_delivery_month"])
    )
)

res = pd.merge(env_training, corr_training, on="aircraft_id", how="inner")


print(res)
