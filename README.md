# sia-tp3
Perceptrón Simple y Multicapa, Lineal como no lineal


# Presentación
La presentación se encuentra [acá](https://docs.google.com/presentation/d/1eYy0SBYM656LqigkSseMVhJhor3rvZayd9KMcy967bU/edit?usp=sharing).


# Archivo de configuración
El archivo de configuración tiene varios campos


### Data
```
x            ruta donde se encuentra el *.txt con los datos de entrada
y            ruta donde se encuentra el *.txt con los datos de salida
perc_train   porcentaje de valores utilizados para entrenar
perc_test    porcentaje de valores utilizados para testear el modelo

```

### Architecture
```
layers              listas de diccionario con campos:
                        activation_function <linear, sigmoid, cosine>
                        dims_in
                        dims_out

w_distribution      nombre de la distribución a utilizar <normal, uniform> para los weights
w_kwargs            argumentos libres que se pasan a la función de numpy correspondiente (ej. w_loc, w_scale para normal)

b_distribution      nombre de la distribución a utilizar <normal, uniform> para los biases
b_kwargs            argumentos libres que se pasan a la función de numpy correspondiente (ej. b_loc, b_scale para normal)
```

### Optimizer
```
name        Nombre del optimizador <AdaGrad, Adam, SGD, Momentum, RMSprop>
kwargs      Argumentos del optimizador
```

### Train
```
epochs      Cantidad de epocas
batch_size  Cantidad de elementos a pasar en forward antes de actualizar el gradiente
```

### Output
```
file        Nombre de archivo de salida . SIN EXTENSION, ya que se guardan los resultados en *.csv y la red en *.pickle
```

# Ejemplo de uso
```python
from src.nn import NN, SGD, Layer

x, y = dataset()
l1 = Layer(35, 10, activation_function="sigmoid")
l2 = Layer(10, 10, activation_function="sigmoid")

optimizer = SGD()
nn = NN([l1, l2], optimizer)
results = nn.train(x, y)
```

