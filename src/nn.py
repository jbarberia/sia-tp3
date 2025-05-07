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
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=1000, batch_size=1, k_fold=None):
        if k_fold:
            X = np.array(x_train.to_list() + x_val.to_list())
            Y = np.array(y_train.to_list() + y_val.to_list())
            indices = [[i for i in range(X.shape[0])] for k in range(k_fold)]
    
        history = []
        for epoch in range(epochs):
            t0 = time()
            if k_fold:
                idx_val = indices[epoch % k_fold]
                idx_train = sum(indices for i in range(k_fold) if i != epoch % k_fold)
        
                x_train = X[idx_train]
                y_train = Y[idx_train]
                x_val = X[idx_val]
                y_val = Y[idx_val]
        
                self._train_model(x_train, y_train, batch_size)
            else:
                self._train_model(x_train, y_train, batch_size)
            t1 = time()

            # metricas de entrenamiento
            metricas = self._obtener_metricas(x_train, y_train, "ENTRENAMIENTO", epoch, t1 - t0)
            history.append(metricas)
            
            # metricas de validacion
            if not x_val is None:
                metricas = self._obtener_metricas(x_val, y_val, "VALIDACION", epoch)                
                history.append(metricas)

        return pd.concat(history)
    

    def _train_model(self, X, Y, batch_size):
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(batch_size):
            batch_indices = np.random.choice(indices, len(indices) // batch_size).tolist()

            batch_X = X[batch_indices]
            batch_Y = Y[batch_indices]
        
            grads_dw = [np.zeros_like(layer.w) for layer in self.layers]
            grads_db = [np.zeros_like(layer.b) for layer in self.layers]

            # 1 - Forward pass
            batch_loss = 0.0
            for x, y in zip(batch_X, batch_Y):
                prediction = self.forward(x)
                grad = prediction - y
                
                batch_loss += 0.5 * np.sum((prediction - y) ** 2)

                # 2 - Calculate weights and bias grad
                local_grads = []
                for layer in reversed(self.layers):
                    grad, dw, db = layer.backward(grad)
                    local_grads.append((dw, db))

                for idx, (dw, db) in enumerate(reversed(local_grads)):
                    grads_dw[idx] += dw
                    grads_db[idx] += db
            
            # 3 - Backward pass
            for layer, dw_sum, db_sum in zip(self.layers, grads_dw, grads_db):
                self.optimizer.update(layer, dw_sum / batch_size, db_sum / batch_size)


    def _obtener_metricas(self, x, y, instancia, epoch, epoch_time=None):
        train_loss = self._loss(x, y)
        y_true = [self.one_hot_decoding(yi) for yi in y]
        y_pred = self.predict(x)
        metricas = self.get_metrics(y_true, y_pred)
        metricas["TOTAL_LOSS"] = train_loss
        metricas["INSTANCIA"] = instancia
        metricas["EPOCH"] = epoch
        metricas["EPOCH_TIME"] = epoch_time
        return metricas


    def _loss(self, x, y):
        return 0.5 * sum((self.forward(xi) - yi)**2 for xi, yi in zip(x, y)) / len(x)


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
