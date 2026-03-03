"""
TFG: Optimización de Hiperparámetros con Grover
Archivo 04: Análisis Topológico y Estimación de Tiempos (6 Qubits)
Descripción: Extrae la profundidad geométrica y el cuello de botella de transpilación.
"""
import json
import os
from qiskit import QuantumCircuit, transpile

# Reconstruimos la función base para medirla
def construir_circuito_logico(n_qubits, estados_marcados):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    
    oracle = QuantumCircuit(n_qubits)
    for estado in estados_marcados:
        for i, bit in enumerate(reversed(estado)):
            if bit == '0': oracle.x(i)
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)
        for i, bit in enumerate(reversed(estado)):
            if bit == '0': oracle.x(i)
            
    diffuser = QuantumCircuit(n_qubits)
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
    # Solo aplicamos una iteración para ver el tamaño base
    return qc

def ejecutar_analisis():
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_script, 'datos_hpo.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)
        
    estados_marcados = [k for k, v in datos["resultados_precision"].items() if v >= 85.0]
    qc_ideal = construir_circuito_logico(6, estados_marcados)
    
    print("=========================================================")
    print(" ANÁLISIS DE COMPLEJIDAD TOPOLÓGICA (Métricas para TFG) ")
    print("=========================================================")
    
    # 1. Circuito Ideal
    ops_ideal = qc_ideal.count_ops()
    print(f"\n--- CIRCUITO LÓGICO (IDEAL) ---")
    print(f"Profundidad teórica: {qc_ideal.depth()}")
    print(f"Total puertas teóricas: {sum(ops_ideal.values())}")
    print(f"Puertas Multi-Controladas (MCX): {ops_ideal.get('mcx', 0)}")
    
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
    print(f"-> Puertas CX (Entrelazamiento): {cx_count}  <--- ¡AQUÍ ESTÁ EL PROBLEMA!")
    
    # 3. Matemática de la Decadencia
    # Si cada CX tiene un 98% de probabilidad de éxito (2% de fallo)...
    prob_supervivencia = (0.98 ** cx_count) * 100
    print(f"\nProbabilidad matemática de que el circuito sobreviva sin errores fatales: ~{prob_supervivencia:.6f}%")
    
    # 4. Tiempos QPU
    t_cx_ns = 100 # ns por CX
    t_sq_ns = 30  # ns por single qubit
    tiempo_shot_us = ((cx_count * t_cx_ns) + (sq_count * t_sq_ns)) / 1000
    tiempo_total_ms = (tiempo_shot_us * 1024) / 1000
    
    print(f"\n--- ESTIMACIÓN DE TIEMPO (VENTAJA CUÁNTICA) ---")
    print(f"Tiempo de ejecución en QPU (1024 shots): {tiempo_total_ms:.2f} milisegundos")
    
    # Guardamos las métricas
    metricas = {
        "profundidad_logica": qc_ideal.depth(),
        "profundidad_fisica": qc_fisico.depth(),
        "total_puertas_cx": cx_count,
        "supervivencia_teorica_porcentaje": round(prob_supervivencia, 6),
        "tiempo_qpu_ms": round(tiempo_total_ms, 2)
    }
    datos["metricas_hardware"] = metricas
    
    with open(ruta_json, 'w') as f:
        json.dump(datos, f, indent=4)
    print("\nMétricas topológicas guardadas en el JSON.")

if __name__ == "__main__":
    ejecutar_analisis()