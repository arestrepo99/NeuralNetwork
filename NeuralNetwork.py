import numpy as np
import pyopencl as cl
from Layers import Layer, Reshape
from settings import densecl
from Tensor import Tensor
from Kernel import Kernel
import pickle

class MismatchedDimension(Exception):
    pass

class NullInputShape(Exception):
    pass

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        # Shape 
        if self.layers[0].inputShape is None:
            raise NullInputShape('Input Layer has no defined input shape, pass parameter: inputShape= ')
        for ind in range(1,len(self.layers)):
            if self.layers[ind].inputShape is None:
                self.layers[ind].initiateInput(self.layers[ind-1].outputShape)

            if not self.layers[ind].inputShape == self.layers[ind-1].outputShape:
                err = f'Dimensions {self.layers[ind].inputShape} and {self.layers[ind-1].outputShape} dont match.'
                raise MismatchedDimension(err)
        
        self.inputShape = layers[0].inputShape
        self.outputShape = layers[-1].outputShape
        self.loss = []

    def getLoss(self,Y):
            self.computeError(Y, self.e)
            self.squareError()
            self.meanError()
            return(self.E.get().mean()/np.prod(self.outputShape))

    def allocateMemory(self, batchSize):
        self.batchSize = np.int32(batchSize)
        
        self.e = Tensor((self.batchSize,*self.outputShape))
        self.e2 = Tensor((self.batchSize,*self.outputShape))
        self.E = Tensor((*self.outputShape,))

        [layer.allocateMemory(batchSize) for layer in self.layers]

        self.computeError = Kernel(densecl.computeError, 
            (batchSize*np.prod(self.outputShape),),
            (self.layers[-1].y,))
        self.squareError = Kernel(densecl.squareError, (np.prod((self.batchSize,*self.outputShape)),),
                        (self.e, self.e2))
        self.meanError = Kernel(densecl.meanError, (np.prod(self.outputShape),),
                        (batchSize, np.prod(self.outputShape), self.e2, self.E))


    def save(self, name):
        import os
        if not os.path.isdir('models'):
            os.mkdir('models')
        pickle.dump((self.loss, 
                    [layer.pack() for layer in self.layers]), 
            open('models/'+name+'.pkl', 'wb'))
    
    def load(name):
        loss, packedlayers = pickle.load(open('models/'+name+'.pkl', 'rb'))
        layers = []
        for layertype, params in packedlayers:
            layers.append(layertype.unpack(params))
        model = NeuralNetwork(layers)
        model.loss = loss
        return model

    def predict(self, x):
        ym1 = x
        for layer in self.layers:
            ym1 = layer.forwardPropagate(ym1)
        return ym1

    def gradientDescent(self, X, Y, lrate):
        self.ypred = self.predict(X)
        self.loss.append(self.getLoss(Y))
        sigma = self.e
        for layer in self.layers[::-1]:
            sigma = layer.backwardPropagate(sigma,lrate)
    
    def train(self, x_train, y_train, epochs, lrate, plotfunc):
        numBatches, batchSize = x_train.shape[:2]
        self.allocateMemory(batchSize)
        for _ in range(epochs):
            batches = np.array(range(numBatches))
            np.random.shuffle(batches)
            for batch in batches:
                self.gradientDescent(x_train[batch], y_train[batch], lrate)
                plotfunc(self)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        str = f'\t\t MODEL SUMMARY: \nInput: \t\t {self.inputShape}\n'
        for layer in self.layers:
            str += layer.__str__()
        return str

#https://learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/