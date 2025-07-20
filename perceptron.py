import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

error_history = []

training_inputs = np.array([[0, 0, 1],
                          [1, 1, 1],
                          [1, 0, 1],
                          [0, 1, 1]])

training_outputs = np.array([[0],
                            [1],
                            [1],
                            [0]])

# dont rly need this
np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting synaptic weights: " + str(synaptic_weights))

for i in range(100000):
    input_layer = training_inputs
    output = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - output
    weight_adjustment = error * sigmoid_derivative(output)
    synaptic_weights += np.dot(input_layer.T, weight_adjustment)

    error_history.append(np.sum(error**2))

print("Synaptic wieghts: " + str(synaptic_weights))
print("Outputs: " + str(output))

plt.plot(error_history)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error Reduction Over Time')
plt.show()