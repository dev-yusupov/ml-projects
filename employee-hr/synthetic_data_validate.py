import pandas as pd
import numpy

df = pd.read_csv("synthetic_data.csv")

df = df.dropna(axis=0).reset_index()

df.to_csv("validated_data.csv")