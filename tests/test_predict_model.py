import feather

from src.models.predict_model import predict

model = feather.read_dataframe("../../data/processed/allhypo.test")

predict(model)