import pandas as pd

data_add = "data/augmented_data_flower_26k.csv"
data_out_add = "data/augmented_data_flower_5k.csv"
numZeroData = 2000
numNegData = 1500
numPosData = 1500

df = pd.read_csv(data_add)
zeroDf = df[df.f.abs() <= 1e-3].reset_index().drop('index', axis=1).sample(frac=1)
posDf = df[df.f > 1e-3].reset_index().drop('index', axis=1).sample(frac=1)
negDf = df[df.f < -1e-3].reset_index().drop('index', axis=1).sample(frac=1)

df = pd.concat([zeroDf[:numZeroData], posDf[:numPosData], negDf[:numNegData]]).reset_index().drop('index', axis=1).sample(frac=1)
df.to_csv(data_out_add, index=False)