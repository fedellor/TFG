"""
TFG: Optimización de Hiperparámetros con QAOA
Archivo 02: Simulación Cuántica Ideal (QAOA)
Descripción: Formula el problema HPO como un Hamiltoniano de Ising y lo resuelve 
             con el Algoritmo de Optimización Cuántica Aproximada (QAOA).
"""
import json
import os
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.primitives import StatevectorSampler 
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

def ejecutar_qaoa_ideal(reps=2):
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_script, 'datos_hpo.json')
    
    print(f"Cargando datos clásicos desde: {ruta_json}")
    with open(ruta_json, 'r') as f:
        datos = json.load(f)
        
    resultados = datos["resultados_precision"]
    n_qubits = 6
    num_estados = 2**n_qubits
    
    # ==========================================
    # 1. CONSTRUCCIÓN DEL HAMILTONIANO DIAGONAL
    # ==========================================
    print("\nMapeando precisiones a un Hamiltoniano de Ising...")
    diagonal = np.zeros(num_estados)
    for estado_binario, precision in resultados.items():
        indice = int(estado_binario, 2)
        diagonal[indice] = -precision # QAOA minimiza
        
    matriz_coste = np.diag(diagonal)
    hamiltoniano = SparsePauliOp.from_operator(Operator(matriz_coste))
    print(f"Hamiltoniano construido con éxito ({len(hamiltoniano)} términos Pauli Z).")

    # ==========================================
    # 2. CONFIGURACIÓN DE QAOA
    # ==========================================
    print(f"\nConfigurando QAOA con p={reps} capas (profundidad controlada)...")
    optimizador = COBYLA(maxiter=200, disp=True)
    
    # ¡LA CLAVE TÉCNICA! QAOA requiere un Sampler, no un Estimator
    sampler = StatevectorSampler()
    qaoa = QAOA(sampler=sampler, optimizer=optimizador, reps=reps)
    
    # ==========================================
    # 3. ENTRENAMIENTO HÍBRIDO
    # ==========================================
    print("\nIniciando optimización híbrida (Minimizando la energía)...")
    resultado_qaoa = qaoa.compute_minimum_eigenvalue(hamiltoniano)
    
    print("\n--- RESULTADOS DE LA OPTIMIZACIÓN ---")
    print(f"Energía mínima encontrada: {resultado_qaoa.eigenvalue.real:.2f} (Equivale a ~{-resultado_qaoa.eigenvalue.real:.2f}% acc)")
    print(f"Tiempo de optimización clásica: {resultado_qaoa.optimizer_time:.2f} segundos")
    
    # ==========================================
    # 4. EXTRACCIÓN DEL ESTADO FINAL
    # ==========================================
    print("\nExtrayendo la distribución de probabilidad cuántica final...")
    circuito_optimo = qaoa.ansatz.assign_parameters(resultado_qaoa.optimal_parameters)
    circuito_optimo.measure_all()
    
    job = sampler.run([circuito_optimo], shots=1024)
    counts_qaoa = job.result()[0].data.meas.get_counts()
    
    print("\n--- TOP 5 ESTADOS MÁS PROBABLES (QAOA IDEAL) ---")
    top_5 = sorted(counts_qaoa.items(), key=lambda item: item[1], reverse=True)[:5]
    for estado, repeticiones in top_5:
        acc_real = resultados.get(estado, 0)
        print(f"Estado |{estado}> ({acc_real}% acc clásica): {repeticiones} medidas")
        
    datos["counts_qaoa_ideal"] = counts_qaoa
    with open(ruta_json, 'w') as f:
        json.dump(datos, f, indent=4)
    print("\nDatos cuánticos guardados correctamente en el JSON de QAOA.")

if __name__ == "__main__":
    ejecutar_qaoa_ideal(reps=2)