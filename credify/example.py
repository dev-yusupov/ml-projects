import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

ones = 0
zeros = 0

for i in range(len(df["RESPONSE"])):
    if df["RESPONSE"][i] == 1:
        ones += 1
    else:
        zeros += 1


print(ones)
print(zeros)