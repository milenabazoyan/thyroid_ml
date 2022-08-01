from typing import Tuple

import pandas as pd
from numpy import nan


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


def transform(input_file: str, binary=False) -> None: # typehints
    # Reading and parsing column names from naming file
    names, numeric, boolean = read_names(input_file)

    # Loading raw df
    df = pd.read_csv(input_file, header=None)

    if names:
        df.columns = names

    df = df.drop(['TBG', 'referral source'], axis=1)

    # Removing bad values ('?'), converting other binary values (not boolean)
    df = df.replace({"?": nan})

    # Handle missing values / listwise deletion
    df = df.dropna().reset_index()

    df['sex'] = df['sex'].replace({'F': 1, 'M': 0}).astype(int)

    # Cleaning 'diagnosis' column
    df['diagnosis'] = df['diagnosis'].apply(lambda x: "negative" if x.split('.')[0] == "negative" else "positive") \
        if binary else df['diagnosis'].apply(lambda x: x.split('.')[0])

    # Converting boolean columns
    for col_name in boolean:
        df[col_name] = df[col_name].replace({"t": 1, "f": 0}).astype(int)

    # Converting numeric columns
    for col_name in df.columns:
        df[col_name] = df[col_name].apply(pd.to_numeric, errors='coerce')

    # Saving cleaned df
    df.to_feather(input_file.replace("data/raw", "data/processed"))


if __name__ == '__main__':
    # For train data
    transform("../../data/raw/allhypo.data", binary=True)
    # For test data
    transform("../../data/raw/allhypo.test", binary=True)
