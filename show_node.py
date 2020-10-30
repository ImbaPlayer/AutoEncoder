import pandas as pd


fileName = "/data/sym/one-class-svm/data/mean_of_five/bin-feature/caida-A-50W-5-0.csv"
df = pd.read_csv(fileName)
print(df.shape)