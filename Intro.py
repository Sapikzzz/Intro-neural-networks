# Network with 3 neurons, only 1 layer

import numpy as np

input = [1, 2, 3, 2.5]  # input for network

# Weights for every neuron
# weights1 = [0.2, 0.8, -0.5, 1.0]     # 1 neuron
# weights2 = [0.5, -0.91, 0.26, -0.5]     # 2 neuron
# weights3 = [-0.26, -0.27, 0.17, 0.87]   # 3 neuron

# Weights in list of lists
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# # Bias for every neuron
# bias1 = 2       # 1 neuron
# bias2 = 3       # 2 neuron
# bias3 = 0.5     # 3 neuron

# Biases in list
biases = [2, 3, 0.5]
bias = 2

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

output = np.dot(weights, input) + biases

print(output)

# a = [1,2,3]
# b = [2,3,4]
# dot_product = 1*2 + 2*3 + 3*4
# np.dot(a, b)