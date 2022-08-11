import feather

from src.models.train_model import get_model

pd_df = feather.read_dataframe("../../data/processed/allhypo.data")
get_model(pd_df)
