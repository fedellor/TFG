"""
TFG: Optimización de Hiperparámetros con QAOA
Archivo 05: Visualización de Resultados a Gran Escala (6 Qubits)
Descripción: Genera un gráfico dual para comparar rendimiento clásico y cuántico (QAOA).
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generar_grafico_qaoa():
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_script, 'datos_hpo.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)
        
    resultados = datos["resultados_precision"]
    
    # ORDENAR LOS ESTADOS: De peor a mejor precisión clásica
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x])
    acc_clasica = [resultados[e] for e in estados_ordenados]
    
    # Extraer counts y convertirlos a probabilidad (%)
    c_ideal = datos.get("counts_qaoa_ideal", {e: 0 for e in estados_ordenados})
    c_ruido = datos.get("counts_qaoa_ruido", {e: 0 for e in estados_ordenados})
    
    shots = 1024
    prob_ideal = [(c_ideal.get(e, 0) / shots) * 100 for e in estados_ordenados]
    prob_ruido = [(c_ruido.get(e, 0) / shots) * 100 for e in estados_ordenados]

    # Configuración del lienzo
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(estados_ordenados))
    width = 0.4

    # Barras de probabilidad cuántica
    ax1.bar(x - width/2, prob_ideal, width, label='QAOA Ideal (Teórico)', color='#4c72b0', alpha=0.9)
    ax1.bar(x + width/2, prob_ruido, width, label='QAOA NISQ (Ruido Térmico)', color='#dd8452', alpha=0.8)

    ax1.set_xlabel('Estados Cuánticos (64 Configuraciones de Hiperparámetros)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probabilidad de Medición Cuántica (%)', fontsize=12, fontweight='bold', color='#2b2b2b')
    
    # Ocultar las etiquetas del eje X por claridad visual
    ax1.set_xticks(x)
    ax1.set_xticklabels(['' for _ in x]) 
    ax1.tick_params(axis='x', length=3)
    
    # Eje secundario para la precisión clásica
    ax2 = ax1.twinx()
    ax2.plot(x, acc_clasica, color='#55a868', linewidth=2.5, label='Precisión Red Neuronal (Clásica)')
    
    ax2.set_ylabel('Precisión de Clasificación Clásica (%)', fontsize=12, fontweight='bold', color='#55a868')
    ax2.set_ylim(0, 100)

    # Unir leyendas
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11, framealpha=0.95)

    plt.title('Evaluación de Resiliencia QAOA (6 Qubits / 64 Estados)', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    
    ruta_img = os.path.join(ruta_script, 'grafica_qaoa_tfg.png')
    plt.savefig(ruta_img, dpi=300)
    print(f"¡Gráfico de alta resolución guardado en: {ruta_img}!")
    
    plt.show()

if __name__ == "__main__":
    generar_grafico_qaoa()