import numpy as np
from Tensor import Tensor
from Kernel import Kernel
from settings import program

class Dense:
    def __init__(self, outputShape, activation, inputShape = None ):
        if isinstance(outputShape,int):
            self.outputShape = (np.int32(outputShape),)
        else:
            self.outputShape = tuple(np.int32(i) for i in outputShape)
        self.activation = activation
        if inputShape is not None:
            self.initiateInput(inputShape)
        else:
            self.inputShape = None
        
    def initiateInput(self, inputShape):
        self.inputShape = tuple(np.int32(i) for i in inputShape)
        self.w = Tensor(np.random.randn(*inputShape,*self.outputShape))
        self.b = Tensor(np.random.randn(*self.outputShape))

    def allocateMemory(self, batchSize):
        outputSize, = self.outputShape
        inputSize, = self.inputShape
        batchSize = np.int32(batchSize)
        self.batchSize = batchSize
        self.v = Tensor((batchSize,outputSize))
        self.y = Tensor((batchSize,outputSize))
        self.dphi = Tensor((batchSize,outputSize))
        self.sigma = Tensor((batchSize,inputSize))
        self.dw = Tensor((batchSize,inputSize,outputSize))
        self.db = Tensor((batchSize,outputSize))

        self.GPUForwardPropagate = Kernel(program.forwardPropagate,
            (batchSize,outputSize),
            (inputSize, outputSize, self.v, self.w, self.b))
        self.activate = Kernel(self.activation(program), 
            (batchSize*outputSize,),
            (outputSize, self.v, self.y, self.dphi))
        self.computedb = Kernel(program.computedb, 
            (batchSize*outputSize,),
            (self.dphi, self.db))
        self.computeGradients = Kernel(program.computeGradients, 
            (batchSize, inputSize, outputSize),
            (inputSize, outputSize, self.dw, self.db))
        self.computeLocalGradient = Kernel(program.computeLocalGradient, 
            (batchSize, inputSize),
            (inputSize, outputSize, self.sigma, self.db, self.dphi, self.w))
        self.learningRule = Kernel(program.learningRule, 
            (inputSize, outputSize),
            (inputSize, outputSize, batchSize, self.dw,self.db,self.w,self.b))
        
    def forwardPropagate(self, ym1):
        self.ym1 = ym1
        self.GPUForwardPropagate(self.ym1)
        self.activate()
        return self.y
    
    def backwardPropagate(self, sigma, lrate):
        self.computedb(sigma)
        self.computeGradients(self.ym1)
        self.computeLocalGradient()
        self.learningRule(np.float32(lrate))
        return self.sigma
        


class Reshape:
    def __init__(self, outputShape, inputShape = None):
        if isinstance(outputShape,int):
            self.outputShape = (np.int32(outputShape),)
        else:
            self.outputShape = outputShape
        if inputShape is not None:
            self.initiateInput(inputShape)
        else:
            self.inputShape = None
        
    def initiateInput(self, inputShape):
        self.inputShape = inputShape

    def allocateMemory(self, batchSize):
        self.batchSize = batchSize
        pass

    def forwardPropagate(self, ym1):
        self.ym1 = ym1
        self.y = self.ym1.reshape((self.batchSize,*self.outputShape))
        return self.y
    
    def backwardPropagate(self, sigma ,lrate):
        self.sigma = sigma.reshape((self.batchSize,*self.inputShape))
        return self.sigma