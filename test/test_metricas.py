import numpy as np
import pandas as pd
from src.nn import NN, SGD, Layer, Momentum
from src.metricas import metricas_one_vs_all, metricas_one_vs_other

def test_metrica_xor():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    nn = NN([l1, l2])
    nn.optimizer = SGD()

    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([[1], [1], [-1], [-1]])

    nn.train(x, y, epochs=3000)

    y_hat = np.round(nn.predict(x)).astype(int)
    metricas = pd.DataFrame(metricas_one_vs_all(y, y_hat)).set_index('CLASE')
    assert (metricas.ACCURACY == 1.0).any()
    

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


def test_metricas_ej3():
    x, y = dataset_ej3()
    l1 = Layer(35, 10, activation_function="sigmoid")
    l2 = Layer(10, 10, activation_function="sigmoid")

    optimizer = SGD()
    nn = NN([l1, l2], optimizer)
    results = nn.train(x, y)

    y_true = np.argmax(y, axis=0).reshape((-1, 1))
    y_hat = np.argmax(nn.predict(x), axis=0).reshape((-1, 1))

    metricas_ova = pd.DataFrame(metricas_one_vs_all(y_true, y_hat)).set_index('CLASE')
    assert metricas_ova.ACCURACY.mean() > 0.7

    metricas_ovo = pd.DataFrame(metricas_one_vs_other(y_true, y_hat))
    metricas_ovo.CLASE = metricas_ovo.CLASE.map(lambda x: int(x))
    metricas_ovo.CLASE_VS = metricas_ovo.CLASE_VS.map(lambda x: int(x))

    assert metricas_ovo.ACCURACY.mean() > 0.7
