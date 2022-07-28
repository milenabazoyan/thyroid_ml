import feather as feather
import numpy as np
import pandas as pd


def removenumber(column):
    for n in column:
        n = n.split('.')[0]
        return n


def to_numeric(df):
    for col in df.columns:
        df[col] = df[col].apply(pd.to_numeric, errors='coerce')
    return df


def makena_dropna(df):
    df = df.replace({"?": np.NAN})
    df = df.dropna(inplace = True)
    return df


def replacing(df):
    df['sex'] = df['sex'].replace({'F': 1, 'M': 0})
    df = df.replace({"t": 1, "f": 0})
    return df


if __name__ == "__main__":
    df = feather.read_dataframe("data/raw/allhypo.data")
    to_nan_results = makena_dropna(df)



