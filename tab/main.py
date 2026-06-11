import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

env_training = pd.read_csv("../data/environment_training.csv")
corr_training = pd.read_csv("../data/corrosions_training.csv")



dates = corr_training["observation_date"]


res = pd.merge(env_training, corr_training, on="aircraft_id", how="inner")



print(res)
