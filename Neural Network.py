import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data



nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.1*np.random.randn(n_inputs, n_neurons)
        self.bias= np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs= inputs
        self.output = np.dot(inputs, self.weights) + self.bias 
    def backward(self, dvalues):
        self.dweights= np.dot(self.inputs.T, dvalues)
        self.dbias= np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs= np.dot(dvalues, self.weights.T)

class Activiation_ReLU:
    def forward(self, inputs):
        self.inputs= inputs
        self.output= np.maximum(0,inputs)
    def backward(self, dvalues):
        self.dinputs= dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs= inputs
        exp_values= np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabilities= exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output= probabilities
    def backward(self, dvalues):
        self.dinputs= np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output= single_output.reshape(-1,1)
            jacobian_matrix= np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index]= np.dot(jacobian_matrix,single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses= self.forward(output, y)
        data_loss= np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples= len(y_pred)
        y_pred_clipped= np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods= -np.log(correct_confidences)

        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples= len(dvalues)
        labels= len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs= -y_true / dvalues
        self.dinputs= self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation= Activation_Softmax()
        self.loss= Loss_CategoricalCrossentropy()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output= self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples= len(dvalues)

        if len(y_true.shape) == 2:
            y_true= np.argmax(y_true, axis=1)
        
        self.dinputs= dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs= self.dinputs/samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate= learning_rate
        self.current_learning_rate= learning_rate
        self.decay= decay
        self.iterations= 0
        self.momentum= momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay*self.iterations))
    
    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentum'):
                layer.weight_momentum= np.zeros_like(layer.weights)
                layer.bias_momentum= np.zeros_like(layer.bias)

            weight_updates= self.momentum * layer.weight_momentum - self.current_learning_rate * layer.dweights
            layer.weight_momentum= weight_updates

            bias_updates= self.momentum * layer.bias_momentum - self.current_learning_rate * layer.dbias
            layer.bias_momentum= bias_updates
        
        else:
            weight_updates= -self.current_learning_rate * layer.dweights
            bias_updates= -self.current_learning_rate * layer.dbias

        layer.weights += weight_updates
        layer.bias += bias_updates

    def post_update_params(self):
        self.iterations += 1

       
class Optimizer_Adagrad:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate= learning_rate
        self.current_learning_rate= learning_rate
        self.decay= decay
        self.iterations= 0
        self.epsilon= epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate= self.learning_rate * (1/(1+self.decay*self.iterations))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache= np.zeros_like(layer.weights)
            layer.bias_cache= np.zeros_like(layer.bias)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbias ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias += -self.current_learning_rate * layer.dbias / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.02, decay=1e-5, epsilon=1e-7, rho=0.999):
        self.learning_rate= learning_rate
        self.current_learning_rate= learning_rate
        self.decay= decay
        self.iterations= 0
        self.epsilon= epsilon
        self.rho= rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1./(1.+self.decay*self.iterations))
    
    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache= np.zeros_like(layer.weights)
            layer.bias_cache= np.zeros_like(layer.bias)

        layer.weight_cache= self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache= self.rho * layer.bias_cache + (1-self.rho) * layer.dbias**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias += -self.current_learning_rate * layer.dbias / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate= learning_rate
        self.current_learning_rate= learning_rate
        self.decay= decay
        self.epsilon= epsilon
        self.iterations= 0
        self.beta_1= beta_1
        self.beta_2= beta_2
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate= self.learning_rate * (1/(1+self.decay*self.iterations))
        
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums= np.zeros_like(layer.weights)
            layer.weight_cache= np.zeros_like(layer.weights)
            layer.bias_momentums= np.zeros_like(layer.bias)
            layer.bias_cache= np.zeros_like(layer.bias)
        
        layer.weight_momentums= self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
        layer.bias_momentums= self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbias
        
        weight_momentums_corrected= layer.weight_momentums / (1-self.beta_1**(self.iterations+1))
        bias_momentums_corrected= layer.bias_momentums / (1-self.beta_1**(self.iterations+1))

        layer.weight_cache= self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights**2
        layer.bias_cache= self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbias**2

        weight_cache_corrected= layer.weight_cache / (1-self.beta_2**(self.iterations+1))
        bias_cache_corrected= layer.bias_cache / (1-self.beta_2**(self.iterations+1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.bias += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
        

        



X,Y= spiral_data(samples=100, classes=3)

dense1= Layer_Dense(2,64)
activation1= Activiation_ReLU()

dense2= Layer_Dense(64,3)
loss_activation= Activation_Softmax_Loss_CategoricalCrossEntropy()

optimizer= Optimizer_SGD(decay=1e-2, momentum=0.9)

for epoch in range(10001):

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    loss= loss_activation.forward(dense2.output, Y)

    predictions= np.argmax(loss_activation.output, axis=1)
    if len(Y.shape) == 2:
        Y= np.argmax(Y, axis=1)
    acurracy= np.mean(predictions==Y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {acurracy:.3f}, ' + f'loss: {loss:.3f},' + f'lr: {optimizer.current_learning_rate}')


    loss_activation.backward(loss_activation.output, Y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


    



#########################################################################################################
'''
Optimizers:

SGD - Stochastic Gradient Descent:
Learning Rate= 1
Parameters - LearningRate*ParametersGradients

SGDM - Stochastic Gradient Descent with Momentum:
Weight_Updates= self.momentum * layer.weight_momentum - self.current_learning_rate * layer.dweights
layer.weight_momentum= weight_updates
layer.weight += weight_updates

AdaGrad - Adaptative Gradiente:
Cache += Parm_gradient ** 2
parm_updates= learning_rate * parm_gradient / (sqrt(cache) + eps)


RMSProp - Root Mean Square Propagation:
Cache = rho * cache + (1-rho) * gradient ** 2
parm_updates= learning_rate * parm_gradient / (sqrt(cache) + eps)


Adam - Adaptative Momentum:


'''