"""
TFG: Optimización de Hiperparámetros con VQE
Archivo 02: Simulación Cuántica Ideal (VQE sobre Hamiltoniano de Pauli)
Descripción: Minimiza el paisaje de energía usando un Ansatz Hardware-Efficient.
"""
import json
import os
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.primitives import StatevectorEstimator, StatevectorSampler 
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

def ejecutar_vqe_ideal(reps=2):
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_script, 'datos_hpo.json')
    
    print(f"Cargando datos clásicos desde: {ruta_json}")
    with open(ruta_json, 'r') as f:
        datos = json.load(f)
        
    resultados = datos["resultados_precision"]
    n_qubits = 6
    num_estados = 2**n_qubits
    
    # ==========================================
    # 1. CONSTRUCCIÓN DEL HAMILTONIANO DE PAULI
    # ==========================================
    print("\nMapeando precisiones a un Hamiltoniano de Pauli (Ising)...")
    diagonal = np.zeros(num_estados)
    
    for estado_binario, precision in resultados.items():
        indice = int(estado_binario, 2)
        # VQE minimiza, así que buscamos la energía más negativa (mayor precisión)
        diagonal[indice] = -precision
        
    matriz_coste = np.diag(diagonal)
    hamiltoniano = SparsePauliOp.from_operator(Operator(matriz_coste))
    print(f"Hamiltoniano construido con éxito ({len(hamiltoniano)} términos Pauli Z).")

    # ==========================================
    # 2. CONFIGURACIÓN DE VQE (Hardware-Efficient Ansatz)
    # ==========================================
    print(f"\nConfigurando VQE con Ansatz 'RealAmplitudes' (reps={reps})...")
    # Este circuito es muy poco profundo y resistente al ruido
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=reps)
    
    optimizador = COBYLA(maxiter=200, disp=True)
    estimador = StatevectorEstimator()
    
    # Instanciamos VQE
    vqe = VQE(estimator=estimador, ansatz=ansatz, optimizer=optimizador)
    
    # ==========================================
    # 3. ENTRENAMIENTO HÍBRIDO
    # ==========================================
    print("\nIniciando optimización híbrida (Minimizando la energía)...")
    resultado_vqe = vqe.compute_minimum_eigenvalue(hamiltoniano)
    
    print("\n--- RESULTADOS DE LA OPTIMIZACIÓN ---")
    print(f"Energía mínima encontrada: {resultado_vqe.eigenvalue.real:.2f} (Equivale a ~{-resultado_vqe.eigenvalue.real:.2f}% de precisión)")
    print(f"Tiempo de optimización clásica: {resultado_vqe.optimizer_time:.2f} segundos")
    print(f"Evaluaciones del circuito cuántico: {resultado_vqe.cost_function_evals}")
    
    # ==========================================
    # 4. EXTRACCIÓN DEL ESTADO FINAL (MEDICIÓN)
    # ==========================================
    print("\nExtrayendo la distribución de probabilidad cuántica final...")
    circuito_optimo = ansatz.assign_parameters(resultado_vqe.optimal_parameters)
    circuito_optimo.measure_all()
    
    sampler = StatevectorSampler()
    job = sampler.run([circuito_optimo], shots=1024)
    
    counts_vqe = job.result()[0].data.meas.get_counts()
    
    print("\n--- TOP 5 ESTADOS MÁS PROBABLES (VQE IDEAL) ---")
    top_5 = sorted(counts_vqe.items(), key=lambda item: item[1], reverse=True)[:5]
    for estado, repeticiones in top_5:
        acc_real = resultados.get(estado, 0)
        print(f"Estado |{estado}> ({acc_real}% acc clásica): {repeticiones} medidas")
        
    # ==========================================
    # 5. GUARDADO DE DATOS
    # ==========================================
    datos["counts_vqe_ideal"] = counts_vqe
    with open(ruta_json, 'w') as f:
        json.dump(datos, f, indent=4)
    print("\nDatos cuánticos guardados correctamente en el JSON de VQE.")

if __name__ == "__main__":
    ejecutar_vqe_ideal(reps=2)