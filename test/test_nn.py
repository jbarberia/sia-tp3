
import numpy as np
from src.nn import NN, SGD, Layer, Momentum
import os

def test_xor():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    nn = NN([l1, l2])
    nn.optimizer = SGD()

    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    nn.train(x, y, epochs=3000)

    # Extrae los datos como numero entero del predictor
    eval = lambda i: np.round(nn.forward(x[i])).astype(int)[0] 

    assert y[0] == eval(0)
    assert y[1] == eval(1)
    assert y[2] == eval(2)
    assert y[3] == eval(3)


def test_xor_with_momentum():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    
    nn = NN([l1, l2], optimizer=Momentum())

    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    nn.train(x, y, epochs=3000)

    # Extrae los datos como numero entero del predictor
    eval = lambda i: np.round(nn.forward(x[i])).astype(int)[0] 

    assert y[0] == eval(0)
    assert y[1] == eval(1)
    assert y[2] == eval(2)
    assert y[3] == eval(3)
    

def test_pickle():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    n1 = NN([l1, l2])

    n1.save("prueba.pickle")
    n2 = NN.load("prueba.pickle")
    os.remove("prueba.pickle")

    assert (n1.layers[0].w == n2.layers[0].w).all
    assert (n1.layers[0].b == n2.layers[0].b).all
