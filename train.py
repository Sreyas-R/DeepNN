from netneural.tensor import Tensor
from netneural.nn import NeuralNet
from netneural.loss import Loss , MSE
from netneural.optim import Optimizer , SGD
from netneural.data import DataIterator , BatchIterator

def train(net : NeuralNet , inputs : Tensor , targets: Tensor , num_epochs: int = 5000 , 
          Iterator : DataIterator = BatchIterator() ,
          loss : Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in Iterator(inputs , targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted , batch.targets)

            grad = loss.grad(predicted , batch.targets)
            net.backward(grad)

            optimizer.step(net)

            print((epoch , epoch_loss))

          