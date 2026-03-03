"""
TFG: Optimización de Hiperparámetros con QAOA
Archivo 03: Simulación Cuántica con Ruido (Hardware NISQ)
Descripción: Evalúa la resiliencia del circuito variacional de QAOA.
"""
import json
import os
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Operator

def ejecutar_qaoa_ruido():
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_script, 'datos_hpo.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)
        
    resultados = datos["resultados_precision"]
    n_qubits = 6
    
    # 1. Reconstruimos el Hamiltoniano para el Ansatz
    diagonal = np.zeros(2**n_qubits)
    for estado_binario, precision in resultados.items():
        diagonal[int(estado_binario, 2)] = -precision
    hamiltoniano = SparsePauliOp.from_operator(Operator(np.diag(diagonal)))
    
    # 2. Generamos el circuito QAOA (Ansatz)
    print("Generando circuito QAOA (p=2 capas)...")
    ansatz = QAOAAnsatz(cost_operator=hamiltoniano, reps=2)
    
    # Asignamos parámetros óptimos simulados
    np.random.seed(42)
    parametros_simulados = np.random.rand(ansatz.num_parameters) * np.pi
    circuito_optimo = ansatz.assign_parameters(parametros_simulados)
    circuito_optimo.measure_all()
    
    # 3. Configurar Modelo de Ruido
    print("\nInyectando modelo de ruido de despolarización (NISQ)...")
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.01, 1)
    error_2q = depolarizing_error(0.02, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'sx', 'rz'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
    
    simulador = AerSimulator(noise_model=noise_model)
    
    # 4. Transpilación a hardware de IBM
    basis_gates = ['x', 'sx', 'rz', 'cx']
    print("Transpilando circuito QAOA...")
    circuito_transpilado = transpile(circuito_optimo, simulador, basis_gates=basis_gates, optimization_level=3)
    
    # 5. Ejecución
    print("Ejecutando simulación con ruido...")
    job = simulador.run(circuito_transpilado, shots=1024)
    counts_ruido = job.result().get_counts()
    
    print("\n--- TOP 10 ESTADOS MÁS PROBABLES (QAOA CON RUIDO) ---")
    top_10 = sorted(counts_ruido.items(), key=lambda item: item[1], reverse=True)[:10]
    for estado, repeticiones in top_10:
        acc_real = resultados.get(estado, 0)
        print(f"Estado |{estado}> ({acc_real}% acc clásica): {repeticiones} medidas")
        
    datos["counts_qaoa_ruido"] = counts_ruido
    with open(ruta_json, 'w') as f:
        json.dump(datos, f, indent=4)
    print("\nDatos con ruido guardados exitosamente.")

if __name__ == "__main__":
    ejecutar_qaoa_ruido()