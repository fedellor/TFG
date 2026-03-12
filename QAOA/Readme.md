# 🕸️ QAOA: Optimización Aproximada y Cuellos de Botella

![Status](https://img.shields.io/badge/Status-En_proceso-orange)

Esta carpeta evalúa el **QAOA (Quantum Approximate Optimization Algorithm)**, un algoritmo variacional diseñado originalmente para problemas de optimización combinatoria en grafos (ej. *MaxCut*).

## 🎯 Enfoque Metodológico
El problema de selección de hiperparámetros (Learning Rate, Batch, Hidden Layers) carece de una topología geométrica predecible. En esta investigación, forzamos al algoritmo a optimizar un Hamiltoniano de Coste diagonal denso compuesto por los 64 estados de precisión clásica para evaluar su eficiencia frente a VQE y Grover.

## ⚙️ Pipeline de Ejecución

1. **`01_benchmark_clasico.py`**: Generación del *ground truth* (espacio de búsqueda).
2. **`02_qaoa_ideal.py`**: Simulación híbrida perfecta utilizando la primitiva `StatevectorSampler` y capas de evolución unitaria ($p=2$).
3. **`03_qaoa_ruido.py`**: Inyección de ruido de despolarización en la arquitectura transpilada de QAOA.
4. **`04_analisis_metricas.py`**: Análisis crítico del cuello de botella topológico. La falta de estructura de grafo en el problema HPO obliga al compilador a insertar una avalancha de puertas SWAP, resultando en **228 puertas CX**. Esto hunde la probabilidad de supervivencia al **~0.99%**.
5. **`05_graficas_qaoa.py`**: Generación de la gráfica de doble eje, revelando visualmente el colapso del circuito bajo ruido real.

## 📊 Artefactos y Conclusiones
* Los datos almacenados en `datos_hpo.json` prueban que, para paisajes de energía desconectados como la Optimización de Hiperparámetros, **QAOA sufre una penalización geométrica letal** durante la transpilación, haciéndolo inviable en hardware NISQ en comparación directa con el enfoque eficiente de VQE.
