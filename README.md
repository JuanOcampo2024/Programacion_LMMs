#Juan Pablo Ocampo Soto - jpablo.ocampo1@udea.edu.co

# Machine Learning Data Preparation Challenge

Este repositorio contiene una serie de retos de programación en **Python** enfocados en el preprocesamiento avanzado de datos utilizando `pandas`, `numpy` y `scikit-learn`.

## Descripción de los Ejercicios

El proyecto se divide en 4 módulos independientes, cada uno abordando un problema del mundo real:

| Ejercicio | Dominio | Tecnologías Clave | Concepto Principal |
| :--- | :--- | :--- | :--- |
| **01. Manufactura** | Industrial | `PolynomialFeatures`, `VarianceThreshold` | Ingeniería de features y filtrado de varianza. |
| **02. Vitivinícola** | Bodega de Vinos | `OrdinalEncoder`, `SelectKBest` | Manejo de variables ordinales y selección estadística. |
| **03. Banca Digital** | Créditos | `ColumnTransformer`, `IsolationForest` | Procesamiento mixto y detección de anomalías. |
| **04. Energía** | Consumo Eléctrico | `SplineTransformer`, `TransformedTargetRegressor` | Series temporales cíclicas y regresión log-normal. |

---

## Estructura de los Scripts

Cada archivo `.py` sigue una estructura estricta para facilitar la evaluación:

1.  **El Problema**: Contexto de negocio.
2.  **La Misión**: Requisitos técnicos de la función `preparar_datos`.
3.  **Solución**: Implementación robusta con `sklearn`.
4.  **Generador de Casos**: Función `generar_caso_de_uso_preparar_datos()` que crea datos sintéticos y el *Ground Truth* para validación automática.

---
