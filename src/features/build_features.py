from pathlib import Path
from typing import Tuple

import pandas as pd
from numpy import nan
from data import raw, processed


def read_names(path) -> Tuple[list, list, list]:
    names = []
    numeric = []
    boolean = []

    names_file_path = str(path).rsplit('.', 1)[0] + ".names"
    with open(names_file_path) as names_f:
        for ln in names_f.readlines():
            if ':' in ln:
                col_name = ln.split(':', 1)[0]
                names.append(col_name)
                if 'continuous' in ln:
                    numeric.append(col_name)
                elif 'f, t' in ln:
                    boolean.append(col_name)
        names.append('diagnosis')
    return names, numeric, boolean


def transform(input_file: Path, binary=False) -> None:
    # Reading and parsing column names from naming file
    names, numeric, boolean = read_names(raw)

    # Loading raw df
    df = pd.read_csv(raw, header=None)

    if names:
        df.columns = names

    # Removing bad values ('?'), converting other binary values (not boolean)
    df = df.replace({"?": nan})
    df['sex'] = df['sex'].replace({'F': 1, 'M': 0}).astype(int)

    # Cleaning 'diagnosis' column
    df['diagnosis'] = df['diagnosis'].apply(lambda x: "negative" if x.split('.')[0] == "negative" else "positive") \
        if binary else df['diagnosis'].apply(lambda x: x.split('.')[0])

    # Converting numeric columns
    for col_name in df.columns:
        df[col_name] = df[col_name].apply(pd.to_numeric, errors='coerce')

    # Converting boolean columns
    for col_name in boolean:
        df[col_name] = df[col_name].replace({"t": 1, "f": 0}).astype(int)

    df = df.drop(['TBG', 'referral source'], axis=1)

    # Handle missing values
    df = df.dropna().reset_index()

    # Saving cleaned df
    df.to_feather(processed)


if __name__ == '__main__':
    # For train data
    transform("allhypo.data")
    # For test data
    transform("allhypo.test")