# 🔍 Algoritmo de Grover: Búsqueda Exacta y Barrera NISQ

![Status](https://img.shields.io/badge/Status-En_proceso-orange)

Esta carpeta contiene la implementación del **Algoritmo de Grover** adaptado para la Optimización de Hiperparámetros (HPO) mediante el enfoque *Grover Adaptive Search* (GAS).

## ⚙️ Pipeline de Ejecución (Flujo de 5 Pasos)

El ecosistema de código está estructurado de forma secuencial para contrastar la teoría cuántica con las limitaciones del hardware actual:

1. **`01_benchmark_clasico.py`**: Entrenamiento clásico (PyTorch/GPU) de 64 configuraciones de hiperparámetros para establecer la línea base.
2. **`02_grover_ideal.py`**: Simulación algorítmica pura (Statevector). Demuestra que el Oráculo dinámico concentra la probabilidad en las soluciones óptimas (precisión > 85%) en solo 2 iteraciones.
3. **`03_grover_ruido.py`**: Simulación de hardware físico. Inyecta el modelo de ruido térmico y de despolarización característico de la era NISQ de IBM.
4. **`04_analisis_metricas.py`**: Radiografía topológica. Transpila el circuito ideal a compuertas físicas para extraer la profundidad y el recuento de puertas entrelazadas (CX), justificando matemáticamente la caída de rendimiento.
5. **`05_graficas_tfg.py`**: Generación visual de los resultados (doble eje) ordenados por rendimiento clásico.

## 📊 Artefactos Generados
* **`datos_hpo.json`**: Diccionario central que almacena las precisiones de la red neuronal, los conteos de medición cuántica y las métricas topológicas de supervivencia.
* **Gráficas (`.png`)**: Representaciones visuales que demuestran el colapso térmico del Oráculo de Grover al escalar la complejidad a 6 Qubits.
