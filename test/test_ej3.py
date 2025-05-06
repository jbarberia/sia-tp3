


import numpy as np
import pandas as pd

from src.nn import NN, SGD, Layer


def dataset_ej3():
    # Generacion del dataset
    data = pd.read_csv("data/TP3-ej3-digitos.txt", header=None, sep=" ").dropna(axis=1)
    data = data.to_numpy()
    x = []
    y = []
    numero = 0
    for i in range(0, data.shape[0], 7):
        x.append(data[i:i+7].reshape(-1))
        out = np.zeros(10)
        out[numero] = 1
        y.append(out)
        numero += 1
    return np.array(x), np.array(y)


def test_ej3():
    x, y = dataset_ej3()
    l1 = Layer(35, 10, activation_function="sigmoid")
    l2 = Layer(10, 10, activation_function="sigmoid")

    optimizer = SGD()
    nn = NN([l1, l2], optimizer)

    results = nn.train(x, y)

    
    acierto = 0
    for (xi, yi) in zip(x, y):
        y_hat = np.argmax(nn.forward(xi))
        yi = np.argmax(yi)

        if y_hat == yi:
            acierto += 1
