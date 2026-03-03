"""
TFG: Optimización de Hiperparámetros con Grover
Archivo 02: Simulación Cuántica Ideal (6 Qubits)
Descripción: Lee el JSON clásico y aplica Grover Adaptive Search ideal.
"""
import json
import math
import os
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

# ==========================================
# 1. CONSTRUCCIÓN DINÁMICA DE GROVER
# ==========================================
def construir_circuito_grover(n_qubits, estados_marcados):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits)) # Superposición uniforme inicial
    
    # Si no hay estados que superen el umbral, devolvemos el circuito base
    if not estados_marcados:
        qc.measure_all()
        return qc, 0

    # Oráculo Dinámico
    oracle = QuantumCircuit(n_qubits, name="Oráculo HPO")
    for estado in estados_marcados:
        for i, bit in enumerate(reversed(estado)):
            if bit == '0': oracle.x(i)
        
        # Puerta MCX (Multi-Controlled X)
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)
        
        for i, bit in enumerate(reversed(estado)):
            if bit == '0': oracle.x(i)
            
    # Difusor
    diffuser = QuantumCircuit(n_qubits, name="Difusor")
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)
    
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    # Cálculo matemático del número óptimo de iteraciones
    N = 2 ** n_qubits
    M = len(estados_marcados)
    iteraciones = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))

    # Ensamblaje
    for _ in range(iteraciones):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
        
    qc.measure_all()
    return qc, iteraciones

# ==========================================
# 2. EJECUCIÓN DEL SIMULADOR
# ==========================================
def ejecutar_simulacion_ideal(umbral=85.0):
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_script, 'datos_hpo.json')
    
    print(f"Cargando datos clásicos desde: {ruta_json}")
    with open(ruta_json, 'r') as f:
        datos = json.load(f)
    
    resultados = datos["resultados_precision"]
    
    # Filtramos las "agujas en el pajar"
    estados_marcados = [k for k, v in resultados.items() if v >= umbral]
    
    print(f"\n--- CONFIGURACIÓN DEL ORÁCULO ---")
    print(f"Umbral de precisión exigido: {umbral}%")
    print(f"Estados válidos (Soluciones): {len(estados_marcados)} de 64")
    print(f"Estados marcados: {estados_marcados}")
    
    if not estados_marcados:
        print("¡Error! Ningún estado supera el umbral. Baja el umbral para continuar.")
        return

    qc, iteraciones = construir_circuito_grover(6, estados_marcados)
    print(f"\nIteraciones cuánticas aplicadas: {iteraciones}")
    
    print("\nSimulando circuito cuántico ideal de 6 qubits (Statevector)...")
    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=1024)
    counts = job.result()[0].data.meas.get_counts()
    
    print("\n--- RESULTADOS IDEALES (Top 5 más medidos) ---")
    top_5 = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
    for estado, repeticiones in top_5:
        precision = resultados.get(estado, 0)
        marcador = " <--- ¡SOLUCIÓN!" if estado in estados_marcados else ""
        print(f"Estado |{estado}> ({precision}% acc): {repeticiones} medidas{marcador}")
        
    # Guardar conteos para usarlos en el gráfico final
    datos["counts_ideal"] = counts
    with open(ruta_json, 'w') as f:
        json.dump(datos, f, indent=4)
        
    print(f"\nDatos cuánticos guardados correctamente en el JSON.")

if __name__ == "__main__":
    ejecutar_simulacion_ideal(umbral=85.0) # Ajustamos el umbral a 85% para filtrar solo a los mejores