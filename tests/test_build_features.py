import feather

pd_df = feather.read_dataframe("../data/raw/allhypo.data")
print(pd_df.head(5))
