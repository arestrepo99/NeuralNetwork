
import numpy as np
from Tensor import Tensor
from Kernel import Kernel
from settings import program

class Dense:
    def __init__(self, inputShape, outputShape, activation):
        self.inputShape = np.int32(inputShape)
        self.outputShape = np.int32(outputShape)
        self.activation = activation

        self.w = Tensor(np.random.randn(inputShape,outputShape))
        self.b = Tensor(np.random.randn(outputShape))
        
    def allocateMemory(self, batchSize):
        outputShape = self.outputShape
        inputShape = self.inputShape
        batchSize = np.int32(batchSize)
        self.batchSize = batchSize
        self.v = Tensor((batchSize,outputShape))
        self.y = Tensor((batchSize,outputShape))
        self.dphi = Tensor((batchSize,outputShape))
        self.sigma = Tensor((batchSize,inputShape))
        self.dw = Tensor((batchSize,inputShape,outputShape))
        self.db = Tensor((batchSize,outputShape))

        self.GPUForwardPropagate = Kernel(program.forwardPropagate,
            (batchSize,outputShape),
            (inputShape, outputShape, self.v, self.w, self.b))
        self.activate = Kernel(self.activation(program), 
            (batchSize*outputShape,),
            (outputShape, self.v, self.y, self.dphi))
        self.computeError = Kernel(program.computeError, 
            (batchSize*outputShape,),
            (self.y, self.dphi, self.db))
        self.computedb = Kernel(program.computedb, 
            (batchSize*outputShape,),
            (self.dphi, self.db))
        self.computeGradients = Kernel(program.computeGradients, 
            (batchSize, inputShape, outputShape),
            (inputShape, outputShape, self.dw, self.db))
        self.computeLocalGradient = Kernel(program.computeLocalGradient, 
            (batchSize, self.inputShape),
            (self.inputShape, self.outputShape, self.sigma, self.db, self.dphi, self.w))
        self.learningRule = Kernel(program.learningRule, 
            (inputShape, outputShape),
            (inputShape, outputShape, batchSize, self.dw,self.db,self.w,self.b))
        
    def forwardPropagate(self, ym1):
        self.ym1 = ym1
        #assert self.ym1.shape[1:] == self.inputShape
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
        