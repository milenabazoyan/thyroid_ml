# 1. Typehints  +
# 2. separate x_test, y_test
# 3. load model from joblib.load and run model.predict on x_test
# 4. see the differences of y_test and model predicted values
# 5. change neighbors to 15 and see if our model is better
# 6. read something about metrics


import joblib

def predict(test_data):
    X_test = test_data.iloc[:, :-1].values
    Y_test = test_data['diagnosis'].values

    model_path = "models/08_30_2022_13_50_00/knn.bin"
    loaded_model = joblib.load(model_path)
    prdct = loaded_model.predict(X_test)