import numpy as np
from Tensor import Tensor
from Kernel import Kernel
from settings import programConv

class Conv:
    def __init__(self, kernel, filters, padding, inputShape , strides, activation):
        self.kernel = np.int32(kernel)
        self.filters = np.int32(filters)
        self.padding = 0#np.int32(padding)
        self.inputShape = tuple(np.int32(i) for i in inputShape)
        self.strides = tuple(np.int32(i) for i in strides)

        self.activation = activation
        self.outputShape = (*tuple((size+2*padding-kernel[dim])//strides[dim] 
                        for dim,size in enumerate(inputShape[:2])),filters)
        self.w = Tensor(np.random.randn(*(filters, *kernel, inputShape[2])))
        self.b = Tensor(np.random.randn(filters))

    def allocateMemory(self, batchSize):
        self.batchSize = np.int32(batchSize)
        self.batchSize = batchSize
        self.v = Tensor((self.batchSize,*self.outputShape))
        self.y = Tensor((self.batchSize,*self.outputShape))
        self.dphi = Tensor((self.batchSize,*self.outputShape))
        self.sigma = Tensor((self.batchSize,*self.inputShape))#Missing something
        self.dw = Tensor((batchSize, self.filters, *self.kernel, self.inputShape[2]))
        self.db = Tensor((batchSize, self.filters))
        
        self.GPUForwardPropagate = Kernel(programConv.forwardPropagate,
            (batchSize, self.filters, np.prod(self.outputShape[:2])),
            (*self.outputShape[:2], self.filters,
             *self.strides, *self.kernel, *self.inputShape,
             self.v, self.w, self.b))
        self.activate = Kernel(self.activation(programConv), 
             (np.prod((batchSize,*self.outputShape)),),
             (self.v, self.y, self.dphi))
        self.computeError = Kernel(programConv.computeError, 
             (np.prod((batchSize,*self.outputShape)),),
             (self.y,))
        self.computedb = Kernel(programConv.computedb, 
             (batchSize,self.filters),
             (*self.outputShape,self.dphi,self.db))
        self.computeGradients = Kernel(programConv.computeGradients, 
             (batchSize*self.filters,np.prod(self.kernel),self.inputShape[2]),
             (*self.outputShape,*self.inputShape,*self.kernel, *self.strides,
             self.dw,self.dphi))
        self.computeLocalGradient = Kernel(programConv.computeLocalGradient, 
            (batchSize*self.filters, np.prod(self.inputShape[:2]), self.inputShape[2]),
            (*self.outputShape,*self.inputShape,*self.kernel,
            self.sigma, self.db, self.w))
        #self.learningRule = Kernel(programConv.learningRule, 
        #    (1, 1),
        #    (0, 0, 0, self.dw,self.db,self.w,self.b))

    def forwardPropagate(self, ym1):
        self.ym1 = ym1
        #assert self.ym1.shape[1:] == self.inputShape
        self.GPUForwardPropagate(self.ym1)
        self.activate()
        return self.y
    
    def backwardPropagate(self, **kwargs):
        if 'Y' in kwargs:
            self.computeError(kwargs['Y'], kwargs['e'])
            self.computedb(kwargs['e'])
            self.computeGradients(self.ym1, kwargs['e'])
        if 'sigma' in kwargs:
            self.computedb(kwargs['sigma'])
            self.computeGradients(self.ym1, kwargs['sigma'])
        self.computeLocalGradient()
        #self.learningRule(np.float32(kwargs['lrate']))
        return self.sigma
        