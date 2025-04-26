import numpy as np
from src.nn import NN, AdaGrad, Adam, Layer, Momentum, RMSprop


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


def test_xor_with_adagrad():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    
    nn = NN([l1, l2], optimizer=AdaGrad())

    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    nn.train(x, y, epochs=2000)

    # Extrae los datos como numero entero del predictor
    eval = lambda i: np.round(nn.forward(x[i])).astype(int)[0] 

    assert y[0] == eval(0)
    assert y[1] == eval(1)
    assert y[2] == eval(2)
    assert y[3] == eval(3)


def test_xor_with_rms_prop():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    
    nn = NN([l1, l2], optimizer=RMSprop())

    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    nn.train(x, y, epochs=3000)

    # Extrae los datos como numero entero del predictor
    eval = lambda i: np.round(nn.forward(x[i])).astype(int)[0] 

    assert y[0] == eval(0)
    assert y[1] == eval(1)
    assert y[2] == eval(2)
    assert y[3] == eval(3)


def test_xor_with_adam():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 1, activation_function="linear")
    
    nn = NN([l1, l2], optimizer=Adam())

    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    nn.train(x, y, epochs=3000)

    # Extrae los datos como numero entero del predictor
    eval = lambda i: np.round(nn.forward(x[i])).astype(int)[0] 

    assert y[0] == eval(0)
    assert y[1] == eval(1)
    assert y[2] == eval(2)
    assert y[3] == eval(3)
