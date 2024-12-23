import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X,y = spiral_data(100,3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.1*np.random.randn(n_inputs, n_neurons)
        self.bias= np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

class Activiation_ReLU:
    def forward(self,inputs):
        self.output= np.maximum(0,inputs)

layer1= Layer_Dense(2, 5)
activation1= Activiation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)


'''
inputs= [[1, 2, 3, 2.5], [2.0,5.0,-1.0,2.0], [-1.5, 2.7, 3.3, -0.8]]
weights= [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
bias= [2,3,0.5]

weights2= [[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]
bias2= [-1,2,-0.5]

layer1_output= np.dot(inputs,np.array(weights).T) + bias

layer2_output= np.dot(layer1_output,np.array(weights2).T) + bias2

print(layer2_output)
'''

'''
for neuron_weights, neuron_bias in zip(weights, bias):
    neuron_output= 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)
'''