# Algorithm 1 training - KNN Neighbors
# https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

# 1. split processed dataframe to x_train and y_train
# 2. use KNN Classifier to fit and get model, use https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# 3. dump model to models/{datetime now}/knn.bin, use "from joblib import dump"

from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
import datetime

def get_model(df):

    X_train = df.iloc[:, :-1]
    Y_train = df['diagnosis']

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, Y_train)

    predictiton = knn_model.predict(X_train)

    x = datetime.datetime.now()

    dump(knn_model, '../models/x')



