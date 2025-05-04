import os
import sys
import json
import tomli
import numpy as np
import pandas as pd

from src.nn import NN, Layer
from src.nn import AdaGrad, Adam, SGD, Momentum, RMSprop



OPTIMIZER = {
    "AdaGrad": AdaGrad,
    "Adam": Adam,
    "SGD": SGD,
    "Momentum": Momentum,
    "RMSprop": RMSprop,
}

DISTRIBUTIONS = {
    "normal": np.random.normal,
    "uniform": np.random.uniform
}


def create_nn(config):
    # crea el optimizador
    optimizer_config = config["optimizer"]
    optimizer = OPTIMIZER[optimizer_config["name"]]()
    for k, v in optimizer_config.items():
        setattr(optimizer, k, v)

    # crea la arquitectura
    layers = []
    
    architecture_config = config["architecture"]
    w_distribution = architecture_config.get("w_distribution")
    b_distribution = architecture_config.get("b_distribution")
    w_options = {k[2:]: v for k, v in architecture_config.items() if k.startswith("w_") and k != "w_distribution"}
    b_options = {k[2:]: v for k, v in architecture_config.items() if k.startswith("b_") and k != "b_distribution"}
    
    for i, options in enumerate(architecture_config["layers"]):
        l = Layer(**options)
        
        # modifico la inicializacion de los pesos
        if w_distribution:
            fun = DISTRIBUTIONS[w_distribution]
            new_w = fun(size=l.w.shape, **w_options)
            l.w = new_w
        
        if b_distribution:
            fun = DISTRIBUTIONS[b_distribution]
            new_b = fun(size=l.b.shape, **b_options)
            l.b = new_b
        
        layers.append(l)

    # ensamblo red
    nn = NN(layers, optimizer)
    return nn


def get_dataset(config):
    x = np.loadtxt(config["data"]["x"])
    y = np.loadtxt(config["data"]["y"]).astype(int)
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    return x, y


def one_hot_encoding(y):
    val2idx = {val: i for i, val in enumerate(np.unique(y))}
    idx2val = {i: val for i, val in enumerate(np.unique(y))}
    dims = y.shape[0], len(val2idx)
    output = np.zeros(dims)

    for i, y_i in enumerate(y):
        output[i, val2idx[y_i]] = 1
    
    return idx2val, output


def train_test_validation_split(x, y, perc_train, perc_test):
    n = x.shape[0]
    n_train = np.floor(perc_train * n).astype(int)
    n_test  = np.floor(perc_test * n).astype(int)
    n_val   = n - n_train - n_test
    
    # mezclo los indices del dataset
    indices = np.arange(n)
    np.random.shuffle(indices)

    # genero las particiones
    x_train, x_test, x_val = x[:n_train], x[n_train:-n_val], x[-n_val:]
    y_train, y_test, y_val = y[:n_train], y[n_train:-n_val], y[-n_val:]
    
    return x_train, y_train, x_test, y_test, x_val, y_val


def main(config):
    # obtiene dataset
    x, y = get_dataset(config)
    idx2class, y = one_hot_encoding(y)

    # particiones para entrenamiento, testeo y validacion
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_validation_split(
        x, y, config["data"]["perc_train"], config["data"]["perc_test"]
    )

    # crea arquitectura
    nn = create_nn(config)
    nn.idx2class = idx2class

    # entrena red
    train_options = config["train"]
    train_results = nn.train(x_train, y_train, x_val, y_val, **train_options)

    # obtiene predicciones con datos nuevos
    y_true = [nn.one_hot_decoding(yi) for yi in y_test]
    y_pred = nn.predict(x_test)

    # genera metricas    
    metricas = nn.get_metrics(y_true, y_pred)
    metricas["INSTANCIA"] = "TEST"

    # devuelve resultados
    results = pd.concat((train_results, metricas))
    return nn, results


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        config = tomli.load(f)

    for i in range(config["corridas"]):
        print("corrida {}".format(i))
        nn, results = main(config)
        
        path = config["output"]["folder"]
        if not os.path.exists(path):
            os.mkdir(path)
        
        
        results.to_csv(os.path.join(path, "corrida_{}.csv".format(i)), index=False)
        nn.save(os.path.join(path, "corrida_{}.pickle".format(i))) 
