#Neural net = made of layers
#Each layer passes inputs forward and propagate gradients backwards
#INput -> linear -> tanh -> linear -> output
from typing import Dict , Callable
from netneural.tensor import Tensor
import numpy as np


class Layer:
    def __init__(self) -> None:
        self. params : Dict[str , Tensor] = {}
        self.grads : Dict[str , Tensor] = {}

    def forward(self , inputs: Tensor) -> Tensor :
        """
        Produce outputs corresponding to these inputs
        """
        raise NotImplementedError
    
    def backward(self , grad: Tensor) -> Tensor:
        #
        """
        Backpropagate this gradient thorugh the layer
        """
        raise NotImplementedError
    

class Linear(Layer):
    #y=f(Wx+b)  W = weight , b = bias
    def __init__(self , input_size:int , output_size:int) ->None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size , output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self , inputs : Tensor) -> Tensor:
        # Outputs = inputs @ w + b

        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self , grad: Tensor) -> Tensor:
        self.grads["b"] = np.sum(grad , axis = 0)       # Gradient of bias 
        self.grads["w"] = self.inputs.T @ grad          #Gradient of weight
        return grad @ self.params["w"].T                 #Backpropagate gradient
    


F = Callable[[Tensor] ,Tensor]

class Activation(Layer):
    #Activiation layer applies a function elementwise to its inputs

    def __init__(self , f: F , f_prime : F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
    
    def forward(self , inputs : Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backwords(self , grad : Tensor) -> Tensor:
        return  self.f_prim(self.innputs) * grad 



def tanh(x : Tensor) -> Tensor:
    return np.tanh(x)
def tanh_prime(x:Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y**2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh , tanh_prime)