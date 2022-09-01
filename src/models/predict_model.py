# 1. Typehints  +
# 2. separate x_test, y_test
# 3. load model from joblib.load and run model.predict on x_test
# 4. see the differences of y_test and model predicted values
# 5. change neighbors to 15 and see if our model is better
# 6. read something about metrics


import joblib
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score


def predict(test_data):
    X_test = test_data.iloc[:, :-1].values
    Y_test = test_data['diagnosis'].values

    model_path = "models/09_01_2022_15_38_53/knn.bin"
    loaded_model = joblib.load(model_path)
    predicted_values = loaded_model.predict(X_test)
    scores = loaded_model.predict_proba(X_test)

    cm = confusion_matrix(Y_test, predicted_values)
    ba = balanced_accuracy_score(Y_test, predicted_values)
    # a = accuracy_score(Y_test, predicted_values)

    # f1 score, matthews_corrcoef and other metrics
    print(predicted_values)
    print(scores)
