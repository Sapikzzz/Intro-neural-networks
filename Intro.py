# Network with 3 neurons, only 1 layer

import numpy as np

inputs = [[1, 2, 3, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]  # batch of input

# Weights for every neuron
# weights1 = [0.2, 0.8, -0.5, 1.0]     # 1 neuron
# weights2 = [0.5, -0.91, 0.26, -0.5]     # 2 neuron
# weights3 = [-0.26, -0.27, 0.17, 0.87]   # 3 neuron

# Weights in list of lists
weights1 = [[0.2, 0.8, -0.5, 1.0],  # weights of first layer with 3 neurons, 4 weight because 4 inputs
           [0.5, -0.91, 0.26, -0.5],    # first subset for first neuron, second for second
           [-0.26, -0.27, 0.17, 0.87]]

weights2 = [[0.1, -0.14, 0.5],  # weights of second layer with 3 neurons
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

# # Bias for every neuron
# bias1 = 2       # 1 neuron
# bias2 = 3       # 2 neuron
# bias3 = 0.5     # 3 neuron

# Biases in list
biases1 = [2, 3, 0.5]
biases2 = [-1, 2, -0.5]


# output = [input[0]*weights1[0]+input[1]*weights1[1]+input[2]*weights1[2]+input[3]*weights1[3]+bias1,
#           input[0]*weights2[0]+input[1]*weights2[1]+input[2]*weights2[2]+input[3]*weights2[3]+bias2,
#           input[0]*weights3[0]+input[1]*weights3[1]+input[2]*weights3[2]+input[3]*weights3[3]+bias3,]

layer_output = []

# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0   # Value for every neuron, zero at the beginning of loop because it goes to the
#                         # next neuron with every iteration
#     for n_input, weight in zip(input, neuron_weights):
#         neuron_output += n_input * weight
#     neuron_output += neuron_bias
#     layer_output.append(neuron_output)

# result of zip(weights, biases)
# (([0.2, 0.8, -0.5, 1.0], 2), ([0.5, -0.91, 0.26, -0.5], 3), ([-0.26, -0.27, 0.17, 0.87], 0.5))

# result of zip(input, weights[0])
# ((1, 0.2), (2, 0.8), (3, -0.5), (2.5, 1.0))

layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1   # Transpose so we can multiply matrixes

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

# a = [1,2,3]
# b = [2,3,4]
# dot_product = 1*2 + 2*3 + 3*4
# np.dot(a, b)