import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def load_data(path):
    return pd.read_csv(path)


def preprocess(df):
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    
    X = df.drop(columns=["quality"])
    y = df["quality"]
    return X, y


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_PATH = os.path.join(BASE_DIR, "data", "WineQT.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Cargando datos desde:", DATA_PATH)
    print("Cargando modelo desde:", MODEL_PATH)

    # 🔹 Cargar datos
    df = load_data(DATA_PATH)
    X, y = preprocess(df)

    # 🔥 IMPORTANTE: mismo split que entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🔹 Cargar modelo y scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 🔹 Escalar SOLO test
    X_test_scaled = scaler.transform(X_test)

    # 🔹 Predicción
    y_pred = model.predict(X_test_scaled)

    # 🔹 Métricas
    print("\n📊 Reporte de clasificación:\n")
    print(classification_report(y_test, y_pred))

    # 🔹 Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()

    print(f"\n📁 Matriz guardada en: {save_path}")