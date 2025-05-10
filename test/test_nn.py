
import numpy as np
from src.perceptron_multicapa import PerceptronMulticapa, Layer
from src.optimizer import SGD, Momentum
import os


def test_xor():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 2, activation_function="sigmoid")
    nn = PerceptronMulticapa([l1, l2])
    nn.optimizer = SGD()
    nn.optimizer.learning_rate = 0.5

    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    y_enc = nn.one_hot_encoding(y)
    nn.train(x, y_enc, epochs=5000)

    # Extrae los datos como numero entero del predictor
    eval = nn.predict(x)

    assert y[0] == eval[0]
    assert y[1] == eval[1]
    assert y[2] == eval[2]
    assert y[3] == eval[3]


def test_xor_with_momentum():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 2, activation_function="linear")
    
    nn = PerceptronMulticapa([l1, l2], optimizer=Momentum())

    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    y_enc = nn.one_hot_encoding(y)
    nn.train(x, y_enc, epochs=5000)

    # Extrae los datos como numero entero del predictor
    eval = nn.predict(x)

    assert y[0] == eval[0]
    assert y[1] == eval[1]
    assert y[2] == eval[2]
    assert y[3] == eval[3]


def test_xor_with_batch():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 2, activation_function="linear")
    
    nn = PerceptronMulticapa([l1, l2], optimizer=Momentum())

    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    y_enc = nn.one_hot_encoding(y)
    nn.train(x, y_enc, epochs=5000, batch_size=2)

    # Extrae los datos como numero entero del predictor
    eval = nn.predict(x)

    assert y[0] == eval[0]
    assert y[1] == eval[1]
    assert y[2] == eval[2]
    assert y[3] == eval[3]
    

def test_pickle():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    n1 = PerceptronMulticapa([l1, l2])

    n1.save("prueba.pickle")
    n2 = PerceptronMulticapa.load("prueba.pickle")
    os.remove("prueba.pickle")

    assert (n1.layers[0].w == n2.layers[0].w).all
    assert (n1.layers[0].b == n2.layers[0].b).all
