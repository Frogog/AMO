import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def load_test_data():
    X_test = pd.read_csv("test/X.csv").to_numpy()
    y_test = pd.read_csv("test/y.csv").to_numpy().ravel()
    return X_test, y_test


def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy:.2f}")
    print("\nМатрица ошибок")
    print(confusion_matrix(y_test, y_pred))



print("Загрузка тестовых данных")
X_test, y_test = load_test_data()

print("Загрузка модели")
model = load_model()

print("Оценка модели")
evaluate_model(model, X_test, y_test)