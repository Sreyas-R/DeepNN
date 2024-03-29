#Loss Function measures how good our predictions are

from netneural.tensor import Tensor
import numpy as np

class Loss:
    def loss(self ,predicted: Tensor , actual : Tensor) -> float:
        raise NotImplementedError
    
    def grad(self , predicted:Tensor , actual : Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    def loss(self ,predicted: Tensor , actual : Tensor) -> float:
        return np.sum((predicted - actual) ** 2)
    
    def grad(self , predicted:Tensor , actual : Tensor) -> Tensor:
        return 2 * (predicted - actual)
    
    


