# Algorithm 1 training - KNN Neighbors
# https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

# 1. split processed dataframe to x_train and y_train
# 2. use KNN Classifier to fit and get model, use https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# 3. dump model to models/{datetime now}/knn.bin, use "from joblib import dump"
import os

from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
from datetime import datetime


def get_model(df: DataFrame) -> None:

    X_train = df.iloc[:, :-1].values
    Y_train = df['diagnosis'].values

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, Y_train)

    filename = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    os.mkdir(f'../../models/{filename}')
    dump(knn_model, f'../../models/{filename}/knn.bin')

