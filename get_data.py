import pandas as pd

if __name__ == "__main__":
    fileName = "data/bin-caida-A-50W-5-0.csv"
    saveName = "data/bin-test.csv"
    df = pd.read_csv(fileName)
    dfb = df.iloc[0:100]
    print(dfb)
    dfb.to_csv(saveName, index=False)