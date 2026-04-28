import pandas as pd
import joblib
import warnings
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# 📥 Carga
def load_data(path):
    return pd.read_csv(path)


# 🧹 Preprocesamiento
def preprocess(df):
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    
    X = df.drop(columns=["quality"])
    y = df["quality"]
    return X, y


# ⚙️ Split + escala
def split_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# 🤖 Entrenamiento + evaluación REAL
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "RF": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)  # 🔥 aquí estaba el error antes

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report["weighted avg"]["f1-score"]

        results[name] = f1
        trained[name] = model

        print(f"{name} -> Accuracy: {acc:.4f} | F1: {f1:.4f}")

    best_model_name = max(results, key=results.get)
    print("\n🔥 Mejor modelo:", best_model_name)

    return trained[best_model_name], results, best_model_name


# 💾 Guardado
def save_model(model, scaler, base_dir):
    model_path = os.path.join(base_dir, "model.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print("Modelo guardado en:", model_path)
    print("Scaler guardado en:", scaler_path)


# 📊 Gráfica
def plot_results(results):
    names = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(8,5))
    plt.bar(names, scores)
    plt.title("Comparación de modelos (F1-score)")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.show()


# 🚀 MAIN
if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_PATH = os.path.join(BASE_DIR, "data", "WineQT.csv")

    print("Cargando datos desde:", DATA_PATH)

    df = load_data(DATA_PATH)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test, scaler = split_scale(X, y)

    model, results, name = train_models(X_train, X_test, y_train, y_test)

    save_model(model, scaler, BASE_DIR)

    plot_results(results)