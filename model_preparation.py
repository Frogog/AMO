import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
import pickle
import os

def load_data():
    x_train = pd.read_csv("train/X.csv").to_numpy()
    y_train = pd.read_csv("train/y.csv").to_numpy().ravel()

    return x_train, y_train


def train_model(X_train, y_train):
    model = LogisticRegressionCV(l1_ratios=(0,),cv=3)
    model.fit(X_train, y_train)
    print("Лучшие параметры модели: C = ", model.C_)
    return model

def save_model(model):
    if not os.path.isdir('model'):
        os.makedirs('model', exist_ok=True)

    filename = 'model/model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def test_model(model):
    x = pd.read_csv('test/X.csv').to_numpy()
    y = pd.read_csv('test/y.csv').to_numpy().ravel()
    print("Предварительная оценка модели: ", model.score(x, y))


if __name__ == "__main__":
    X_train, y_train = load_data()
    model = train_model(X_train, y_train)
    save_model(model)
    test_model(model)





