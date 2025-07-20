from layer import Layer
from dense import Dense
from activation import Activation
from hyperbolic_tangent import tanh
from error import mse, mse_prime
import numpy as np
import matplotlib.pyplot as plt

X = np.reshape([[0,0],
             [1,0],
             [0,1],
             [1,1]], (4, 2, 1))

Y = np.reshape([[0],
            [1],
            [1],
            [0]], (4, 1, 1))

network = [
    Dense(2,3),
    tanh(),
    Dense(3,1),
    tanh()
]

epochs = 1000
learning_rate = 0.1
error_history = []

for i in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        
        error = mse(y, output)
        error_history.append(error)

        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
        
    error /= len(X)
    if i % 50 == 0:
        print(f'Epoch {i+1}/{epochs}, Error={error}')

plt.plot(error_history)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()