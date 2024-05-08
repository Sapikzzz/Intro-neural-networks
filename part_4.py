import numpy as np

np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons);
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

print(layer1.weights)
layer1.forward(X)       # 0.17640523*1 - 0.09772779*2 + 0.01440436*3 + 0.03336743*2.5 = output for first neuron in first batch (weight for input * input + next weight for input * next input + ...)
# layer2.forward(layer1.output)
#
print(layer1.output)