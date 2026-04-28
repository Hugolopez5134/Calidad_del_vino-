# 🍷 Wine Quality Classification - Proyecto de IA

## 📌 Descripción

Proyecto de clasificación supervisada para predecir la calidad del vino a partir de variables físico-químicas.

Se desarrolló inicialmente en un notebook monolítico y posteriormente se **modularizó en una arquitectura reproducible**, separando EDA, entrenamiento y evaluación.

---

## 🎯 Objetivo

Construir un pipeline de Machine Learning que:

* Analice el dataset (EDA)
* Entrene múltiples modelos
* Compare desempeño
* Guarde el mejor modelo
* Permita su evaluación posterior

---

## 🧠 Tipo de problema

* Clasificación supervisada multiclase
* Variable objetivo: `quality`

---

## 📊 Dataset

* Registros: 1143
* Variables: 13
* No contiene valores nulos
* Dataset desbalanceado (clases medias dominan)

---

## 🏗️ Estructura del proyecto

```
Calidad_del_vino/
│
├── data/
│   └── WineQT.csv
│
├── src/
│   ├── eda.py
│   ├── entrenamiento.py
│   └── prueba.py
│
├── outputs/
│   ├── histogramas.png
│   ├── correlacion.png
│   ├── scatter_clases.png
│   └── confusion_matrix.png
│
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

### 1. Clonar repositorio

```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

### 3. Activar entorno

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución

### 1. Análisis exploratorio (EDA)

```bash
python src/eda.py
```

Genera:

* Histogramas
* Boxplots (outliers)
* Matriz de correlación
* Scatter de separabilidad

---

### 2. Entrenamiento

```bash
python src/entrenamiento.py
```

* Modelos utilizados:

  * Regresión Logística
  * KNN
  * Random Forest

* Métrica principal:

  * F1-score (por desbalance)

Salida:

* Modelo guardado (`model.pkl`)
* Escalador (`scaler.pkl`)
* Comparación de modelos

---

### 3. Evaluación

```bash
python src/prueba.py
```

* Reporte de clasificación
* Matriz de confusión
* Resultados guardados en `/outputs`

---

## 📈 Resultados

| Modelo              | F1-score |
| ------------------- | -------- |
| Logistic Regression | 0.5906   |
| KNN                 | 0.6720   |
| Random Forest       | 0.7035   |

🏆 **Mejor modelo: Random Forest**

---

## 📊 Insights clave

* Dataset sin valores nulos (no se requirió imputación)
* Clases desbalanceadas → uso de F1-score
* Separabilidad parcial entre clases
* Presencia de outliers (no eliminados)

---

## ⚠️ Problemas encontrados

### 1. Error de rutas (FileNotFoundError)

**Causa:** rutas relativas incorrectas
**Solución:** uso de rutas dinámicas con `os.path`
**Estado:** ✅ resuelto

---

### 2. Modelo no encontrado (model.pkl)

**Causa:** no se ejecutó entrenamiento antes de prueba
**Solución:** ejecutar pipeline completo
**Estado:** ✅ resuelto

---

### 3. Evaluación incorrecta (data leakage)

**Causa:** evaluación sobre todo el dataset
**Solución:** uso de `train_test_split` estratificado
**Estado:** ✅ resuelto

---

## 🚀 Mejoras futuras

* Implementar SMOTE para balanceo
* Optimización de hiperparámetros (GridSearch)
* API con FastAPI para inferencia
* Deploy en la nube

---

## 🧑‍💻 Autor

Víctor López
Ingeniero Químico | Estudiante de IA & Data Science

---

## 📌 Notas finales

Proyecto diseñado bajo principios de:

* Modularidad
* Reproducibilidad
* Buenas prácticas en ML

---

*"Si no puedes correrlo en otra máquina, no es un proyecto… es un accidente."*
