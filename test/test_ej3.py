import numpy as np
import pandas as pd

from src.perceptron_multicapa import PerceptronMulticapa, Layer
from src.optimizer import SGD, Adam


def dataset_ej3():
    # Generacion del dataset
    data = pd.read_csv("data/TP3-ej3-digitos.txt", header=None, sep=" ").dropna(axis=1)
    data = data.to_numpy()
    x = []
    y = []
    numero = 0
    for i in range(0, data.shape[0], 7):
        x.append(data[i:i+7].reshape(-1))
        y.append(numero)
        numero += 1
    return np.array(x), np.array(y)


def test_ej3_digitos():
    optimizer = Adam()
    l1 = Layer(35, 10, activation_function="sigmoid")
    l2 = Layer(10, 10, activation_function="sigmoid")
    nn = PerceptronMulticapa([l1, l2], optimizer)
    
    x, y = dataset_ej3()
    y_enc = nn.one_hot_encoding(y)
    
    nn.train(x, y_enc, epochs=1000)

    prediccion = nn.predict(x)
    assert (y == prediccion).all()
    

def test_ej3_paridad():
    optimizer = Adam()
    l1 = Layer(35, 10, activation_function="sigmoid")
    l2 = Layer(10, 2, activation_function="sigmoid")
    nn = PerceptronMulticapa([l1, l2], optimizer)
    
    x, y = dataset_ej3()
    y = y % 2
    y_enc = nn.one_hot_encoding(y)
    
    nn.train(x, y_enc, epochs=1000)

    prediccion = nn.predict(x)
    assert (y == prediccion).all()


def test_ej3_minimo():
    optimizer = Adam()
    l1 = Layer(35, 2, activation_function="linear")
    nn = PerceptronMulticapa([l1], optimizer)
    
    x, y = dataset_ej3()
    y = y % 2
    y_enc = nn.one_hot_encoding(y)
    
    nn.train(x, y_enc, epochs=1000)

    prediccion = nn.predict(x)
    assert (y == prediccion).all()


def test_ej3_kfold():
    optimizer = Adam()
    l1 = Layer(35, 2, activation_function="linear")
    nn = PerceptronMulticapa([l1], optimizer)
    
    x, y = dataset_ej3()
    y = y % 2
    y_enc = nn.one_hot_encoding(y)
    
    nn.train(x, y_enc, epochs=1000, k_fold=3)

    prediccion = nn.predict(x)
    assert (y == prediccion).all()
