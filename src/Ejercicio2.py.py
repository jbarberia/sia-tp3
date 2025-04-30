# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:56:49 2025

@author: ArianSaggese
"""
import numpy as np
import matplotlib.pyplot as plt
# # Entradas
# inputs = np.array([
#     [0,0],
#     [0,1],
#     [1,0],
#     [1,1],
#  ])

# # Salidas
# labels = np.array([0,0,0,1])


# Cargar datos desde línea 5 (índice 4 en Python), separado por coma
datos = np.genfromtxt('TP3-ej2-conjunto.csv', delimiter=',', skip_header=4)

# Separar entradas y salidas
X_x = datos[:, :-1]  # todas las columnas menos la última
Y_y = datos[:, -1]   # última columna

# Porcentaje de datos para entrenamiento
percent_training_population = 0.9
N = int(np.floor(percent_training_population * len(X_x)))

# Conjunto de entrenamiento
inputs = X_x[:N, :]
labels = Y_y[:N]

# Conjunto de validación
X_val = X_x[N:, :]
y_val = Y_y[N:]


# plt.scatter(inputs[:,0],inputs[:,1],marker = 'o')
# plt.show()

N = np.size(inputs[1,:])
learning_rate = 0.05
epochs = 1000

def activation(z):
    return 1 / (1 + np.exp(-z)) 

def train(inputs,labels,learning_rate,epochs):
    # Se inicializan pesos y bias
    weights = np.random.rand(N)
    bias = np.random.rand()
    average_error_save = []
    for epoch in range(epochs):
        error_acumulado = 0
        for input,label in zip(inputs,labels):
            # print(f"input: {input}, label:{label}",end = " ")
            z = np.dot(input,weights) + bias
            y_pred = activation(z)
            
            # Error
            error = activation(label) - y_pred
            # print(f"error = {error} en epoca {epoch}")
            error_acumulado += abs(error)             
            delta_w = learning_rate*error*input
            weights = weights + delta_w
            
            delta_b = learning_rate*error
            bias = bias + delta_b    
        average_error = error_acumulado/len(inputs)
        average_error_save.append(average_error) 
        print(f"Error  :{average_error} en epoca: {epoch} Delta_w : {delta_w}")
    return weights, bias,average_error_save
            

w,b,error = train(inputs, labels, learning_rate, epochs)



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
# plt.plot(np.arange(0,epochs),error)
# plt.title("Error vs Epoca")
# plt.show()
print(f"Pesos ajustados: {w}")
print(f"Bias ajustado: {b}")

# %matplotlib inline
# %matplotlib qt
