# pylande: reportMissingImports=false
import enum
import numpy as np
import pyopencl as cl
import time

from settings import program

from Tensor import Tensor
from Kernel import Kernel

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.inputShape = layers[0].inputShape
        self.outputShape = layers[-1].outputShape
        self.loss = []
        for ind in range(1,len(self.layers)):
            assert self.layers[ind].inputShape == self.layers[ind-1].outputShape

    def getLoss(self,):
            self.squareError()
            self.meanError()
            return(self.E.get().mean())

    def allocateMemory(self, batchSize):
        self.batchSize = np.int32(batchSize)
        
        self.e = Tensor((self.batchSize,*self.outputShape))
        self.e2 = Tensor((self.batchSize,*self.outputShape))
        self.E = Tensor((*self.outputShape,))

        [layer.allocateMemory(batchSize) for layer in self.layers]



        self.squareError = Kernel(program.squareError, (np.prod((self.batchSize,*self.outputShape)),),
                        (self.e, self.e2))
        self.meanError = Kernel(program.meanError, (np.prod(self.outputShape),),
                        (batchSize, np.prod(self.outputShape), self.e2, self.E))
        
        
    
    def predict(self, x):
        ym1 = x
        for layer in self.layers:
            ym1 = layer.forwardPropagate(ym1)
        return ym1
        
    def gradientDescent(self, X, Y, lrate):
        self.ypred = self.predict(X)
        for ind,layer in enumerate(self.layers[::-1]):
            if ind == 0:
                sigma = layer.backwardPropagate(Y = Y, e = self.e, lrate = lrate)
            else:
                sigma = layer.backwardPropagate(sigma = sigma, lrate = lrate)
            
    
    def train(self, x_train, y_train, epochs, lrate, plotfunc):
        numBatches, batchSize, _ = x_train.shape
        self.allocateMemory(batchSize)
        for _ in range(epochs):
            batches = np.array(range(numBatches))
            np.random.shuffle(batches)
            for batch in batches:
                self.gradientDescent(x_train[batch], y_train[batch], lrate)
                self.loss.append(self.getLoss())
                plotfunc(self.loss)
    
    def testGrads(self, X, Y, step = 0.1):
        self.gradientDescent(X, Y, 0)
        dw = []
        db = []
        l = self.getLoss()
        for layer in self.layers:
            dw.append([])
            db.append([])
            from Dense import Reshape
            if isinstance(layer,Reshape):
                continue
            w = layer.w.get().flatten()
            for i in range(w.size):
                w0 = w[i]
                w[i] = w0 + step
                layer.w.set(w)
                self.gradientDescent(X,Y,lrate = 0)
                dw[-1].append((self.getLoss()-l)/step)
                w[i] = w0
                layer.w.set(w)
            b = layer.b.get().flatten()
            for i in range(b.size):
                b0 = b[i]
                b[i] = b0 + step
                layer.b.set(b)
                self.gradientDescent(X,Y,lrate = 0)
                db[-1].append((self.getLoss()-l)/step)
                b[i] = b0
                layer.b.set(b)
        return dw,db


sigmoid = lambda x: x.sigmoid

#https://learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/