# 1. Typehints
# 2. separate x_test, y_test
# 3. load model from joblib.load and run model.predict on x_test
# 4. see the differences of y_test and model predicted values
# 5. change neighbors to 15 and see if our model is better
# 6. read something about metrics


import joblib

def predict(model):
    X_test = model.iloc[:, :-1].values
    Y_test = model['diagnosis'].values

    loaded_model = joblib.load(model)
    prdct = loaded_model.predict(X_test)
