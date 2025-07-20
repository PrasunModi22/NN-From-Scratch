import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        self.error_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, training_inputs, training_outputs, epochs):
        for i in range(epochs):
            input_layer = training_inputs.astype(float)
            self.output = self.sigmoid(np.dot(input_layer, self.synaptic_weights))
            flipped_inputs = np.transpose(input_layer)

            error = training_outputs - self.output
            self.error_history.append(np.sum(error**2))
            weight_adjustments = error * self.sigmoid_derivative(self.output)
            self.synaptic_weights += np.dot(flipped_inputs, weight_adjustments)



if __name__ == "__main__":
    nn = NeuralNetwork()
    training_inputs = np.array([[0, 0, 1],
                          [1, 1, 1],
                          [1, 0, 1],
                          [0, 1, 1]])
    training_outputs = np.array([[0],
                            [1],
                            [1],
                            [0]])
    nn.train(training_inputs, training_outputs, 100000)
    print("Outputs: " + str(nn.output))

    plt.plot(nn.error_history)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error Reduction Over Time')
    plt.show()