"""
TFG: Optimización de Hiperparámetros con Grover
Archivo 03: Simulación Cuántica con Ruido (6 Qubits - Era NISQ)
Descripción: Evalúa el impacto de la decoherencia y errores de puertas físicas.
"""
import json
import math
import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ==========================================
# 1. CONSTRUCTOR DEL CIRCUITO
# ==========================================
def construir_circuito_grover(n_qubits, estados_marcados):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    
    if not estados_marcados:
        qc.measure_all()
        return qc

    oracle = QuantumCircuit(n_qubits, name="Oráculo HPO")
    for estado in estados_marcados:
        for i, bit in enumerate(reversed(estado)):
            if bit == '0': oracle.x(i)
            
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)
        
        for i, bit in enumerate(reversed(estado)):
            if bit == '0': oracle.x(i)
            
    diffuser = QuantumCircuit(n_qubits, name="Difusor")
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    N = 2 ** n_qubits
    M = len(estados_marcados)
    iteraciones = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))

    for _ in range(iteraciones):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
        
    qc.measure_all()
    return qc

# ==========================================
# 2. SIMULACIÓN CON MODELO DE RUIDO
# ==========================================
def ejecutar_simulacion_ruido(umbral=85.0):
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_script, 'datos_hpo.json')
    
    print("Cargando datos del entorno ideal...")
    with open(ruta_json, 'r') as f:
        datos = json.load(f)
        
    resultados = datos["resultados_precision"]
    estados_marcados = [k for k, v in resultados.items() if v >= umbral]
    
    qc = construir_circuito_grover(6, estados_marcados)
    
    print("\nConfigurando hardware cuántico ruidoso (NISQ)...")
    # Modelo conservador: 1% de fallo en puertas de 1 qubit, 2% en las entrelazadas (CNOT)
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.01, 1)
    error_2q = depolarizing_error(0.02, 2)
    
    noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'sx', 'rz'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
    
    simulador = AerSimulator(noise_model=noise_model)
    
    print("Transpilando circuito (Este paso puede tardar un poco por la puerta MCX de 6 qubits)...")
    # Transpilamos a las puertas base reales del IBM Quantum
    basis_gates = ['x', 'sx', 'rz', 'cx']
    circuito_transpilado = transpile(qc, simulador, basis_gates=basis_gates, optimization_level=3)
    
    print("Ejecutando simulación...")
    job = simulador.run(circuito_transpilado, shots=1024)
    counts_ruido = job.result().get_counts()
    
    print("\n--- RESULTADOS CON RUIDO (Top 10 más medidos) ---")
    top_10 = sorted(counts_ruido.items(), key=lambda item: item[1], reverse=True)[:10]
    for estado, repeticiones in top_10:
        precision = resultados.get(estado, 0)
        marcador = " <--- ¡SOLUCIÓN!" if estado in estados_marcados else ""
        print(f"Estado |{estado}> ({precision}% acc): {repeticiones} medidas{marcador}")
        
    # Guardar en JSON
    datos["counts_ruido"] = counts_ruido
    with open(ruta_json, 'w') as f:
        json.dump(datos, f, indent=4)
    print("\nDatos con ruido guardados exitosamente.")

if __name__ == "__main__":
    ejecutar_simulacion_ruido(umbral=85.0)