import numpy as np

def step(x):
    return 1 if x >= 0 else -1


class PerceptronSimple:
    learning_rate = 0.1
    activation = step

    def __init__(self, input_size):
        #self.weights = np.zeros(input_size + 1) # El 1 es por el bias
        self.weights = np.random.random(input_size + 1) # El 1 es por el bias
        
    def predict(self, x):
        x = np.insert(x, 0, 1) # El bias es x_0 * 1.0
        z = np.dot(self.weights, x)
        return step(z)

    def train(self, x, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(x, y):
                xi = np.insert(xi, 0, 1)  # add bias input
                prediction = step(np.dot(self.weights, xi))
                self.weights += 2 * self.learning_rate * (target - prediction) * xi
