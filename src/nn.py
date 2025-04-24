import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)


FUNCTIONS = {
    "sigmoid": sigmoid,
    "linear": linear,
}

DERIVATIVES = {
    "sigmoid": sigmoid_derivative,
    "linear": linear_derivative,
}


class Layer:
    def __init__(self, dims_in, dims_out, activation_function="sigmoid"):
        self.w = np.random.random((dims_out, dims_in))
        self.b = np.random.random(dims_out)
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
    
    def backward(self, grad_out, learning_rate):
        # https://en.wikipedia.org/wiki/Backpropagation
        # delta_j = dz = dE/do * do/dnet = grad_out * diff_activation_fun
        # dWij = -n * delta_j * out_i = -n * dz .* x
        dz = grad_out * self.activation_derivative(self.z) # delta = dL/d0 * dphi/dnet
        
        dw = np.outer(dz, self.x)  
        db = dz                    

        grad_input = self.w.T @ dz # sum(w * dl) * dz
        self.w -= learning_rate * dw
        self.b -= learning_rate * db

        return grad_input
    

class NN:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x, y, learning_rate=0.01):        
        prediction = self.forward(x)
        grad = prediction - y
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)     
        return 0.5 * np.sum((prediction - y) ** 2)
    

    def train(self, X, Y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in zip(X, Y):
                total_loss += self.backward(x, y, learning_rate)


