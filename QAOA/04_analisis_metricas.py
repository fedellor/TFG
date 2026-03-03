"""
TFG: Optimización de Hiperparámetros con QAOA
Archivo 04: Análisis Topológico y Estimación de Tiempos (QAOA)
Descripción: Extrae la profundidad geométrica y el conteo de puertas CX de QAOA.
"""
import json
import os
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Operator

def ejecutar_analisis_qaoa():
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_script, 'datos_hpo.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)
        
    n_qubits = 6
    resultados = datos["resultados_precision"]
    
    print("=========================================================")
    print(" ANÁLISIS DE COMPLEJIDAD TOPOLÓGICA QAOA (Métricas TFG) ")
    print("=========================================================")
    
    # 1. Reconstruimos el Hamiltoniano y el circuito QAOA
    diagonal = np.zeros(2**n_qubits)
    for estado_binario, precision in resultados.items():
        diagonal[int(estado_binario, 2)] = -precision
    hamiltoniano = SparsePauliOp.from_operator(Operator(np.diag(diagonal)))
    
    ansatz = QAOAAnsatz(cost_operator=hamiltoniano, reps=2)
    
    # Asignamos parámetros aleatorios para poder medir su estructura real
    np.random.seed(42)
    parametros = np.random.rand(ansatz.num_parameters)
    qc_ideal = ansatz.assign_parameters(parametros)
    qc_ideal.measure_all()
    
    ops_ideal = qc_ideal.count_ops()
    print(f"\n--- CIRCUITO LÓGICO (IDEAL QAOA) ---")
    print(f"Profundidad teórica: {qc_ideal.depth()}")
    print(f"Total puertas teóricas: {sum(ops_ideal.values())}")
    
    # 2. Transpilación a Hardware Físico Real (IBM Superconductor)
    print("\nTraduciendo al lenguaje máquina cuántico (Transpilación)...")
    basis_gates = ['x', 'sx', 'rz', 'cx']
    qc_fisico = transpile(qc_ideal, basis_gates=basis_gates, optimization_level=3)
    
    ops_fisico = qc_fisico.count_ops()
    cx_count = ops_fisico.get('cx', 0)
    sq_count = sum(ops_fisico.values()) - cx_count
    
    print(f"\n--- CIRCUITO FÍSICO (NISQ REALIDAD) ---")
    print(f"Profundidad real: {qc_fisico.depth()}")
    print(f"Total puertas físicas: {sum(ops_fisico.values())}")
    print(f"-> Puertas CX (Entrelazamiento): {cx_count}  <--- ¡EL DATO CLAVE!")
    
    # 3. Matemática de la Decadencia
    prob_supervivencia = (0.98 ** cx_count) * 100
    print(f"\nProbabilidad matemática de que el circuito sobreviva sin errores fatales: ~{prob_supervivencia:.4f}%")
    
    # 4. Tiempos QPU
    t_cx_ns = 100 # ns por CX
    t_sq_ns = 30  # ns por single qubit
    tiempo_shot_us = ((cx_count * t_cx_ns) + (sq_count * t_sq_ns)) / 1000
    tiempo_total_ms = (tiempo_shot_us * 1024) / 1000
    
    print(f"\n--- ESTIMACIÓN DE TIEMPO (VENTAJA CUÁNTICA QAOA) ---")
    print(f"Tiempo de ejecución en QPU (1024 shots): {tiempo_total_ms:.2f} milisegundos")
    
    # Guardamos las métricas
    metricas = {
        "profundidad_logica": qc_ideal.depth(),
        "profundidad_fisica": qc_fisico.depth(),
        "total_puertas_cx": cx_count,
        "supervivencia_teorica_porcentaje": round(prob_supervivencia, 4),
        "tiempo_qpu_ms": round(tiempo_total_ms, 2)
    }
    datos["metricas_hardware_qaoa"] = metricas
    
    with open(ruta_json, 'w') as f:
        json.dump(datos, f, indent=4)
    print("\nMétricas topológicas guardadas en el JSON.")

if __name__ == "__main__":
    ejecutar_analisis_qaoa()