from dataclasses import dataclass
import pickle
import numpy as np
import pandas as pd
from time import time

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def cosine(x):
    return np.cos(x)

def cosine_derivative(x):
    return - np.sin(x)


FUNCTIONS = {
    "sigmoid": sigmoid,
    "linear": linear,
    "cosine": cosine,
}

DERIVATIVES = {
    "sigmoid": sigmoid_derivative,
    "linear": linear_derivative,
    "cosine": cosine_derivative,
}

# ------------------------------------------------------------------------------
class Optimizer:
    def update(self, layer, dw, db):
        return NotImplemented


class SGD(Optimizer):
    learning_rate = 0.01
    
    def update(self, layer, dw, db):
        layer.w -= self.learning_rate * dw
        layer.b -= self.learning_rate * db


class Momentum(Optimizer):
    learning_rate = 0.01
    momentum = 0.8
    
    def update(self, layer, dw, db):
        prev_update_w = layer.__dict__.get("prev_update_w", 0)
        prev_update_b = layer.__dict__.get("prev_update_b", 0)

        delta_w = self.learning_rate * dw + self.momentum * prev_update_w
        delta_b = self.learning_rate * db + self.momentum * prev_update_b
        
        layer.prev_update_w = delta_w
        layer.prev_update_b = delta_b
        
        layer.w -= delta_w
        layer.b -= delta_b


class AdaGrad(Optimizer):
    learning_rate = 0.01
    epsilon = 1e-8

    def update(self, layer, dw, db):
        velocity_w = layer.__dict__.get("velocity_w", 0)
        velocity_b = layer.__dict__.get("velocity_b", 0)

        layer.velocity_w = velocity_w + dw**2
        layer.velocity_b = velocity_b + db**2

        layer.w -= self.learning_rate / (np.sqrt(layer.velocity_w) + self.epsilon) * dw
        layer.b -= self.learning_rate / (np.sqrt(layer.velocity_b) + self.epsilon) * db


class RMSprop(Optimizer):
    learning_rate = 0.01
    decay_rate = 0.99
    epsilon = 1e-8

    def update(self, layer, dw, db):
        velocity_w = layer.__dict__.get("velocity_w", 0)
        velocity_b = layer.__dict__.get("velocity_b", 0) 

        layer.velocity_w = self.decay_rate * velocity_w + (1 - self.decay_rate) * dw**2
        layer.velocity_b = self.decay_rate * velocity_b + (1 - self.decay_rate) * db**2

        layer.w -= self.learning_rate / (np.sqrt(layer.velocity_w) + self.epsilon) * dw
        layer.b -= self.learning_rate / (np.sqrt(layer.velocity_b) + self.epsilon) * db


class Adam(Optimizer):
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    def update(self, layer, dw, db):
        t = layer.__dict__.get("t", 0) + 1
        layer.t = t

        momentum_w = layer.__dict__.get("momentum_w", 0)
        momentum_b = layer.__dict__.get("momentum_b", 0)

        velocity_w = layer.__dict__.get("velocity_w", 0)
        velocity_b = layer.__dict__.get("velocity_b", 0)

        layer.momentum_w = self.beta1 * momentum_w + (1 - self.beta1) * dw
        layer.momentum_b = self.beta1 * momentum_b + (1 - self.beta1) * db

        layer.velocity_w = self.beta2 * velocity_w + (1 - self.beta2) * dw**2
        layer.velocity_b = self.beta2 * velocity_b + (1 - self.beta2) * db**2
       
        momentum_w_hat = layer.momentum_w / (1 - self.beta1**(t))
        momentum_b_hat = layer.momentum_b / (1 - self.beta1**(t))

        velocity_w_hat = layer.velocity_w / (1 - self.beta2**(t))
        velocity_b_hat = layer.velocity_b / (1 - self.beta2**(t))

        layer.w -= self.learning_rate * momentum_w_hat / (np.sqrt(velocity_w_hat) + self.epsilon)
        layer.b -= self.learning_rate * momentum_b_hat / (np.sqrt(velocity_b_hat) + self.epsilon)


# ------------------------------------------------------------------------------
class Layer:
    def __init__(self, dims_in, dims_out, activation_function="sigmoid"):
        self.w = np.random.randn(dims_out, dims_in)
        self.b = np.zeros(dims_out)
        self.activation = FUNCTIONS[activation_function]
        self.activation_derivative = DERIVATIVES[activation_function]
        self.x = None
        self.z = None
        self.a = None

    def forward(self, x):
        self.x = x
        self.z = self.w @ x + self.b
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, grad_out):
        # https://en.wikipedia.org/wiki/Backpropagation
        # delta_j = dz = dE/do * do/dnet = grad_out * diff_activation_fun
        # dWij = -n * delta_j * out_i = -n * dz .* x
        dz = grad_out * self.activation_derivative(self.z) # delta = dL/d0 * dphi/dnet
        dw = np.outer(dz, self.x)  
        db = dz                    
        grad_input = self.w.T @ dz # sum(w * dl) * dz
        return grad_input, dw, db


class NN:
    def __init__(self, layers, optimizer=None):
        self.layers = layers
        self.optimizer = optimizer if optimizer else SGD()
        self.idx2class = None
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x, y):        
        prediction = self.forward(x)
        grad = prediction - y
        for layer in reversed(self.layers):
            grad, dw, db = layer.backward(grad)
            self.optimizer.update(layer, dw, db)
            
        return 0.5 * np.sum((prediction - y) ** 2)
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=1000, batch_size=1):
        n_samples = len(x_train)
        history = []

        for epoch in range(epochs):
            t0 = time()
            total_loss = 0.0
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size].tolist()
                
                batch_X = x_train[batch_indices]
                batch_Y = y_train[batch_indices]             
            
                grads_dw = [np.zeros_like(layer.w) for layer in self.layers]
                grads_db = [np.zeros_like(layer.b) for layer in self.layers]

                batch_loss = 0.0
                for x, y in zip(batch_X, batch_Y):
                    prediction = self.forward(x)
                    grad = prediction - y
                    
                    batch_loss += 0.5 * np.sum((prediction - y) ** 2)

                    local_grads = []
                    for layer in reversed(self.layers):
                        grad, dw, db = layer.backward(grad)
                        local_grads.append((dw, db))

                    for idx, (dw, db) in enumerate(reversed(local_grads)):
                        grads_dw[idx] += dw
                        grads_db[idx] += db

                    for layer, dw_sum, db_sum in zip(self.layers, grads_dw, grads_db):
                        self.optimizer.update(layer, dw_sum / batch_size, db_sum / batch_size)

            total_loss += batch_loss
            t1 = time()

            train_loss = 0.5 * sum((self.forward(xi) - yi)**2 for xi, yi in zip(x_train, y_train))
            
            y_true = [self.one_hot_decoding(yi) for yi in y_train]
            y_pred = self.predict(x_train)
            metricas = self.get_metrics(y_true, y_pred)
            metricas["TOTAL_LOSS"] = train_loss
            metricas["INSTANCIA"] = "ENTRENAMIENTO"
            metricas["EPOCH"] = epoch
            metricas["EPOCH_TIME"] = t1 - t0
            history.append(metricas)
            
            # validacion
            if x_val:
                val_loss = 0.5 * sum((self.forward(xi) - yi)**2 for xi, yi in zip(x_val, y_val))

                y_true = [self.one_hot_decoding(yi) for yi in y_val]
                y_pred = self.predict(x_val)
                metricas = self.get_metrics(y_true, y_pred)    
                metricas["TOTAL_LOSS"] = val_loss
                metricas["INSTANCIA"] = "VALIDACION"
                history.append(metricas)

        return pd.concat(history)
    

    def predict(self, x):
        pred = []
        for xi in x:
            forw = self.forward(xi)
            indx = np.argmax(forw)
            if self.idx2class:
                pred.append(self.idx2class[indx])
            else:
                pred.append(indx)
        return pred


    def get_metrics(self, y_true, y_pred, classes=None):
        if self.idx2class:
            classes = self.idx2class.values()
        else:
            classes = set(y_true) | set(y_pred)

        metrics = {c: {'N':0, 'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for c in classes}
        for true, pred in zip(y_true, y_pred):
            for c in metrics:
                if true == c:
                    metrics[c]['N'] += 1
                if true == c and pred == c:
                    metrics[c]['TP'] += 1
                elif true != c and pred == c:
                    metrics[c]['FP'] += 1
                elif true == c and pred != c:
                    metrics[c]['FN'] += 1
                elif true != c and pred != c:
                    metrics[c]['TN'] += 1

        df = pd.DataFrame(metrics).T
        df["ACCURACY"]  = (df.TP + df.TN) / (df.TP + df.TN + df.FP + df.FN)
        df["PRECISION"] = df.TP / (df.TP + df.FP)
        df["RECALL"]    = df.TP / (df.TP + df.FN)
        df["F1_SCORE"]  = 2 * df.TP / (2 * df.TP + df.FP + df.FN)
        df.index.name = "CLASE"
        df = df.reset_index()
        
        return df


    def one_hot_decoding(self, x):
        if self.idx2class:
            return self.idx2class[np.argmax(x)]
        else:
            return np.argmax(x)


    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)


    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


    def one_hot_encoding(self, y):
        val2idx = {val: i for i, val in enumerate(np.unique(y))}
        idx2val = {i: val for i, val in enumerate(np.unique(y))}
        dims = y.shape[0], len(val2idx)
        output = np.zeros(dims)

        for i, y_i in enumerate(y):
            output[i, val2idx[y_i]] = 1
        
        self.idx2class = idx2val
        return idx2val, output
