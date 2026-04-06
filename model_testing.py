import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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
    cm = confusion_matrix(y_test, y_pred)
    print(f"Точность модели: {accuracy:.2f}")
    print("\nМатрица ошибок")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp +fn) if (tp +fn) > 0 else 0
    f1 = 2 * precision * recall * (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"\nrecall: {recall:.4f}")
    print(f"\nF1-score: {f1:.4f}")

    print("\n" + "="*50)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Нет гипертонии', 'Гипертония']))



print("Загрузка тестовых данных")
X_test, y_test = load_test_data()

print("Загрузка модели")
model = load_model()

print("Оценка модели")
evaluate_model(model, X_test, y_test)