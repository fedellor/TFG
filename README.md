# ⚛️ TFG: Optimización Cuántica de Hiperparámetros (HPO) en la Era NISQ

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Qiskit](https://img.shields.io/badge/Qiskit-2.x-purple?logo=qiskit)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?logo=pytorch)
![Status](https://img.shields.io/badge/Status-En_proceso-orange)

Este repositorio contiene el código fuente, los datos y los resultados de las simulaciones pertenecientes al **Trabajo de Fin de Grado (TFG)** sobre la aplicación de la Computación Cuántica para la Optimización de Hiperparámetros (HPO) en modelos de *Deep Learning*.

## 📖 Resumen del Proyecto

El entrenamiento de Redes Neuronales requiere la exploración de vastos espacios de hiperparámetros (Learning Rate, Batch Size, Hidden Layers). Este proyecto evalúa empíricamente la viabilidad de sustituir los métodos clásicos (*Grid Search*) por algoritmos cuánticos para acelerar esta búsqueda. 

Se ha diseñado un entorno híbrido (Cuántico-Clásico) codificando 64 configuraciones de hiperparámetros en un registro de **6 Qubits**. El estudio compara exhaustivamente tres de los algoritmos cuánticos más prominentes tanto en entornos ideales (Statevector) como bajo ruido térmico real (Hardware NISQ de IBM):

1. **Grover (Búsqueda Exacta)**
2. **VQE (Variational Quantum Eigensolver - Ansatz Hardware-Efficient)**
3. **QAOA (Quantum Approximate Optimization Algorithm)**

## 📂 Estructura del Repositorio

El proyecto está modularizado en las siguientes carpetas:

* 📁 **`Data/`**: Contiene el subconjunto de datos extraído del dataset **MNIST** utilizado para entrenar y evaluar las redes neuronales clásicas mediante PyTorch.
* 📁 **`Grover/`**: Implementación del Algoritmo de Grover (Oráculo dinámico + Difusor). Incluye el benchmark clásico, simulaciones ideales y con ruido, extracción de métricas topológicas y gráficas de resultados.
* 📁 **`Pauli/` (VQE)**: Implementación variacional mapeando las precisiones a un Hamiltoniano de Pauli Z. Utiliza un Ansatz `RealAmplitudes` altamente resistente al ruido.
* 📁 **`QAOA/`**: Implementación híbrida utilizando un Hamiltoniano de Coste diagonal. Evalúa la capacidad del algoritmo para optimizar paisajes de energía sin estructura geométrica definida.

> **Nota:** Dentro de cada carpeta algorítmica se encuentra un archivo `.json` autogenerado con las métricas crudas de las simulaciones (conteos de medición, profundidad del circuito físico, número de compuertas entrelazadas CX y estimaciones de tiempo en QPU) y una gráfica `.png` comparativa dual de los resultados.

## 🔬 Conclusiones Clave de la Investigación

La extracción de métricas topológicas durante el proceso de transpilación demostró las severas limitaciones del hardware cuántico actual (Era NISQ) y la clara superioridad de los enfoques heurísticos:

* **Colapso de Grover:** La necesidad de compuertas Multi-Controladas (MCX) en el oráculo generó circuitos de **494 puertas CX**, reduciendo la supervivencia térmica teórica al **~0.004%**, resultando en puro ruido blanco en la simulación física.
* **Cuello de Botella en QAOA:** Al mapear un paisaje de energía denso sin correlación geométrica, el Ansatz generó **228 puertas CX**, limitando la supervivencia al **~0.99%**.
* **Resiliencia de VQE (El Ganador Práctico):** Al utilizar un Ansatz adaptado a la topología física (*Hardware-Efficient*), el entrelazamiento se redujo drásticamente a solo **10 puertas CX**. Esto elevó la supervivencia teórica por encima del **81%**, permitiendo al algoritmo encontrar configuraciones subóptimas pero altamente eficientes (superando el 83% de precisión clásica) bajo niveles severos de ruido.

## ⚙️ Instalación y Reproducibilidad

Para reproducir este entorno de investigación en tu máquina local, asegúrate de tener Python 3.11+ instalado e instala las dependencias necesarias:

```bash
git clone https://github.com/fedellor/TFG.git
cd TFG
pip install -r requirements.txt
