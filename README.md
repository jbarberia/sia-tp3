# sia-tp3
Perceptrón Simple y Multicapa, Lineal como no lineal

# Presentación
La presentación se encuentra [acá](https://docs.google.com/presentation/d/1eYy0SBYM656LqigkSseMVhJhor3rvZayd9KMcy967bU/edit?usp=sharing).

# Ejemplo de uso
```python
from src.nn import PerceptronMulticapa, SGD, Layer

x, y = dataset()
l1 = Layer(35, 10, activation_function="sigmoid")
l2 = Layer(10, 10, activation_function="sigmoid")

optimizer = SGD()
nn = PerceptronMulticapa([l1, l2], optimizer)
results = nn.train(x, y)

y_hat = nn.predict(x)
y_hat == y
```

