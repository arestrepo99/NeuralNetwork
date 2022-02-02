
import numpy as np
from Tensor import Tensor
from Kernel import Kernel
from settings import program

class Dense:
    def __init__(self, inputSize, outputSize, activation):
        self.inputShape = (np.int32(inputSize),)
        self.outputShape = (np.int32(outputSize),)
        self.activation = activation

        self.w = Tensor(np.random.randn(inputSize,outputSize))
        self.b = Tensor(np.random.randn(outputSize))
        
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
        self.computeError = Kernel(program.computeError, 
            (batchSize*outputSize,),
            (self.y, self.dphi, self.db))
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
    
    def backwardPropagate(self, **kwargs):
        if 'Y' in kwargs:
            self.computeError(kwargs['Y'], kwargs['e'])
        if 'sigma' in kwargs:
            self.computedb(kwargs['sigma'])
        self.computeGradients(self.ym1)
        self.computeLocalGradient()
        self.learningRule(np.float32(kwargs['lrate']))
        return self.sigma
        