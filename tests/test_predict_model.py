import feather

from src.models.predict_model import predict

test_data = feather.read_dataframe("../../data/processed/allhypo.test")

predict(test_data)
