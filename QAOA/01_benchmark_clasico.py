"""
TFG: Optimización de Hiperparámetros con Grover
Archivo 01: Benchmark Clásico ACELERADO POR GPU (6 Qubits -> 64 Estados)
Descripción: Genera y evalúa un Grid Search de 64 combinaciones usando PyTorch y CUDA.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import itertools
import json
import time
import os

# ==========================================
# 1. CONFIGURACIÓN DE GPU Y DATOS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==================================================")
print(f" MOTOR DE ENTRENAMIENTO: {device.type.upper()}")
print(f"==================================================")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Usamos 2000 imágenes para entrenamiento y 500 para test (rápido pero representativo)
subset_train = Subset(dataset_train, range(2000))
subset_test = Subset(dataset_test, range(500))

# ==========================================
# 2. DEFINICIÓN DE LA RED NEURONAL
# ==========================================
class SimpleFFNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleFFNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(self.flatten(x))))

# ==========================================
# 3. ESPACIO DE BÚSQUEDA (6 QUBITS = 64 ESTADOS)
# ==========================================
# Dividimos los 6 qubits en 3 dimensiones de hiperparámetros (2 qubits por dimensión = 4 valores cada uno)
lr_vals = [0.001, 0.005, 0.01, 0.05]      # 2 qubits
batch_vals = [16, 32, 64, 128]            # 2 qubits
hidden_vals = [16, 32, 64, 128]           # 2 qubits

def decimal_a_binario(n, bits):
    return format(n, f'0{bits}b')

hyperparam_grid = {}
print("Generando el Grid Search de 64 combinaciones...")

for lr, bs, hd in itertools.product(lr_vals, batch_vals, hidden_vals):
    # Formato binario: q5 q4 (Hidden) | q3 q2 (Batch) | q1 q0 (LR)
    estado_binario = decimal_a_binario(hidden_vals.index(hd), 2) + \
                     decimal_a_binario(batch_vals.index(bs), 2) + \
                     decimal_a_binario(lr_vals.index(lr), 2)
                     
    hyperparam_grid[estado_binario] = {"lr": lr, "batch": bs, "hidden": hd, "epochs": 3}

# ==========================================
# 4. EVALUACIÓN ACELERADA EN GPU
# ==========================================
def ejecutar_grid_search_gpu():
    resultados = {}
    tiempo_inicio = time.time()
    total_evals = len(hyperparam_grid)
    
    print(f"\nIniciando entrenamiento masivo ({total_evals} redes neuronales)...")
    
    for i, (estado, params) in enumerate(hyperparam_grid.items()):
        train_loader = DataLoader(subset_train, batch_size=params["batch"], shuffle=True)
        test_loader = DataLoader(subset_test, batch_size=params["batch"], shuffle=False)
        
        # Subimos el modelo a la RTX
        model = SimpleFFNN(params["hidden"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.CrossEntropyLoss()
        
        # Entrenamiento
        model.train()
        for _ in range(params["epochs"]):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device) # Datos a la GPU
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                
        # Evaluación
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = round(100 * correct / total, 2)
        resultados[estado] = accuracy
        
        if (i + 1) % 16 == 0 or (i + 1) == total_evals:
            print(f"Progreso: [{i+1}/{total_evals}] completados. Último estado evaluado |{estado}>: {accuracy}% acc")

    tiempo_total = time.time() - tiempo_inicio
    mejor_estado = max(resultados, key=resultados.get)
    mejor_acc = resultados[mejor_estado]
    
    print(f"\n--- GRID SEARCH CLÁSICO FINALIZADO ---")
    print(f"Tiempo total: {tiempo_total:.2f} segundos")
    print(f"Mejor configuración: {mejor_estado} ({hyperparam_grid[mejor_estado]}) -> {mejor_acc}%")
    
    # Exportar datos a la misma carpeta donde está el script
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_script, 'datos_hpo.json')
    
    datos_exportacion = {
        "resultados_precision": resultados,
        "tiempo_clasico_segundos": round(tiempo_total, 2)
    }
    
    with open(ruta_json, 'w') as f:
        json.dump(datos_exportacion, f, indent=4)
        
    print(f"Datos guardados correctamente en '{ruta_json}'.")

if __name__ == "__main__":
    ejecutar_grid_search_gpu()