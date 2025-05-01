# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:56:49 2025

@author: ArianSaggese
"""
import numpy as np
import matplotlib.pyplot as plt


# Cargar datos desde línea 5 (índice 4 en Python), separado por coma
datos = np.genfromtxt('TP3-ej2-conjunto.csv', delimiter=',', skip_header=4)

# Separar entradas y salidas
X_x = datos[:, :-1]  # todas las columnas menos la última
Y_y = datos[:, -1]   # última columna

# Porcentaje de datos para entrenamiento
percent_training_population = 1
N = int(np.floor(percent_training_population * len(X_x)))

# Conjunto de entrenamiento
inputs = X_x[:N, :]
labels = Y_y[:N]

# Conjunto de validación
X_val = X_x[N:, :]
y_val = Y_y[N:]


# Otro ejemplo

# # Semilla para reproducibilidad
# np.random.seed(42)

# # Grupo 0: centrado en (1, 1)
# group_0 = np.random.randn(50, 2) * 0.5 + np.array([1, 1])
# labels_0 = np.zeros(50)

# # Grupo 1: centrado en (4, 4)
# group_1 = np.random.randn(50, 2) * 0.5 + np.array([4, 4])
# labels_1 = np.ones(50)

# # Unir ambos grupos
# inputs = np.vstack((group_0, group_1))
# labels = np.concatenate((labels_0, labels_1))

N = np.size(inputs[1,:])
learning_rate = 0.005
epochs = 10000

def activation(z):
    return 1 / (1 + np.exp(-z)) 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def train_backprop(inputs, labels, learning_rate, epochs):
    N_features = inputs.shape[1]
    weights = np.random.randn(N_features)
    bias = np.random.randn()
    error_log = []
    predicts = []
    for epoch in range(epochs):
        total_error = 0
        for x, y in zip(inputs, labels):
            z = np.dot(x, weights) + bias
            y_pred = sigmoid(z)
            error = y - y_pred
            total_error += error**2

            # Gradiente de la función de pérdida con respecto a la predicción
            dE_dy = -error  # derivada de (1/2)(y - y_pred)^2

            # Gradiente de la activación
            dy_dz = sigmoid_derivative(z)

            # Gradientes finales
            dz_dw = x
            dz_db = 1

            gradient_w = dE_dy * dy_dz * dz_dw
            gradient_b = dE_dy * dy_dz * dz_db

            # Actualización de parámetros
            weights -= learning_rate * gradient_w
            bias -= learning_rate * gradient_b
        
        mse = total_error / len(inputs)
        error_log.append(mse)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, MSE: {mse}")
            
    y_pred_all = activation(np.dot(inputs, weights) + bias)
    return weights, bias, error_log, y_pred_all


def compute_metrics_manual(y_true, y_pred):
    # (0 o 1)
    y_true = np.round(y_true).astype(int)
    y_pred = np.round(y_pred).astype(int)

    #  contadores
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Matriz de confusión
    conf_matrix = np.array([[TN, FP],
                            [FN, TP]])

    # Métricas
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }



w,b,error,y_pred = train_backprop(inputs, labels, learning_rate, epochs)

metrics = compute_metrics_manual(activation(labels), y_pred)
# Grafico de variables de entrada
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c='black', marker='o')

# Grafico del plano de separación

# Crear grilla para el plano
x_vals, y_vals = np.meshgrid(np.linspace(inputs[:,0].min(), inputs[:,0].max(), 10),
                              np.linspace(inputs[:,1].min(), inputs[:,1].max(), 10))
z_vals = -(w[0]*x_vals + w[1]*y_vals + b) / w[2]
ax.plot_surface(x_vals, y_vals, z_vals, alpha=0.5, color='cyan')

plt.show()

# Gráfico del error
plt.plot(np.arange(0,epochs),error)
plt.title("Error vs Epoca")
plt.show()
print(f"Pesos ajustados: {w}")
print(f"Bias ajustado: {b}")

# Estas lineas visualizan la salida en graficos 3d con Spyder
# %matplotlib inline
# %matplotlib qt

