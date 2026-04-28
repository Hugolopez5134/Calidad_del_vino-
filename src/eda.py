import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")


def load_data(path):
    return pd.read_csv(path)


def basic_eda(df):
    print("Shape:", df.shape)
    print("\nTipos:\n", df.dtypes)
    print("\nNulos:\n", df.isnull().sum())
    print("\nDescripción:\n", df.describe())


def plot_visuals(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 📊 Histogramas
    df.hist(figsize=(12,10))
    plt.suptitle("Distribución de variables")
    plt.savefig(os.path.join(output_dir, "histogramas.png"))
    plt.close()

    # 📦 Boxplots (outliers)
    cols = ["residual sugar", "chlorides", "alcohol"]
    for col in cols:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Outliers en {col}")
        plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"))
        plt.close()

    # 🔥 Correlación
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.savefig(os.path.join(output_dir, "correlacion.png"))
    plt.close()

    # 🎯 Scatter (separabilidad)
    plt.figure()
    sns.scatterplot(
        data=df,
        x="alcohol",
        y="volatile acidity",
        hue="quality",
        palette="viridis"
    )
    plt.title("Separabilidad de clases")
    plt.savefig(os.path.join(output_dir, "scatter_clases.png"))
    plt.close()

    print(f"\nGráficas guardadas en: {output_dir}")


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_PATH = os.path.join(BASE_DIR, "data", "WineQT.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    print("Cargando datos desde:", DATA_PATH)

    df = load_data(DATA_PATH)
    basic_eda(df)
    plot_visuals(df, OUTPUT_DIR)