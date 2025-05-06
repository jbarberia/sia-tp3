import numpy as np
from src.nn import NN, AdaGrad, Adam, Layer, Momentum, RMSprop


def one_hot_encoding(y):
    val2idx = {val: i for i, val in enumerate(np.unique(y))}
    idx2val = {i: val for i, val in enumerate(np.unique(y))}
    dims = y.shape[0], len(val2idx)
    output = np.zeros(dims)

    for i, y_i in enumerate(y):
        output[i, val2idx[y_i]] = 1
    
    return idx2val, output


def eval_xor(nn):
    # mapeo XOR
    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    idx2class, y_enc = one_hot_encoding(y)

    nn.idx2class = idx2class
    nn.train(x, y_enc, epochs=500)

    # Extrae los datos como numero entero del predictor
    eval = nn.predict(x)

    assert y[0] == eval[0]
    assert y[1] == eval[1]
    assert y[2] == eval[2]
    assert y[3] == eval[3]


def test_with_momentum():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 2, activation_function="linear")
    optimizer = Momentum()
    nn = NN([l1, l2], optimizer)
    eval_xor(nn)


def _test_with_adagrad(): # Este no lo vimos en clase
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 2, activation_function="linear")
    optimizer = AdaGrad()
    nn = NN([l1, l2], optimizer)
    eval_xor(nn)


def test_with_rms_prop():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 2, activation_function="linear")
    optimizer = RMSprop()
    nn = NN([l1, l2], optimizer)
    eval_xor(nn)


def test_with_adam():
    l1 = Layer(2, 3, activation_function="sigmoid")
    l2 = Layer(3, 2, activation_function="linear")
    optimizer = Adam()
    optimizer.learning_rate = 0.1
    nn = NN([l1, l2], optimizer)
    eval_xor(nn)
