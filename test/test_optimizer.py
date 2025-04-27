import numpy as np
from src.nn import NN, AdaGrad, Adam, Layer, Momentum, RMSprop


def eval_xor(nn):
    # mapeo XOR
    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    nn.train(x, y, epochs=3000)

    # Extrae los datos como numero entero del predictor
    eval = lambda i: np.round(nn.forward(x[i])).astype(int)[0] 

    assert y[0] == eval(0)
    assert y[1] == eval(1)
    assert y[2] == eval(2)
    assert y[3] == eval(3)


def test_with_momentum():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    optimizer = Momentum()
    nn = NN([l1, l2], optimizer)
    eval_xor(nn)


def test_with_adagrad():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    optimizer = AdaGrad()
    nn = NN([l1, l2], optimizer)
    eval_xor(nn)


def test_with_rms_prop():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    optimizer = RMSprop()
    nn = NN([l1, l2], optimizer)
    eval_xor(nn)


def test_with_adam():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    optimizer = Adam()
    optimizer.learning_rate = 0.1
    nn = NN([l1, l2], optimizer)
    eval_xor(nn)
