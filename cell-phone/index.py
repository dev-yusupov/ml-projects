import pandas as pd

dataset = pd.read_csv("train_data.csv")
pd.set_option("display.max_columns", None)

columns = ['battery_power', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', ]

