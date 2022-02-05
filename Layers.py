import numpy as np
from Tensor import Tensor
from Kernel import Kernel
from settings import program, programConv

class Activation:
    sigmoid = program.sigmoid

class Layer:
    def __str__(self) -> str:
        return f'{self.__name__} Layer: \t\t{self.outputShape}'



class Dense(Layer):
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
        self.activate = Kernel(self.activation, 
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
    
    def unpack(initiateParams,w,b):
        outputShape, activation, inputShape = initiateParams
        instance = Dense(outputShape, activation, inputShape= inputShape)
        instance.w.set(w)
        instance.b.set(b)

    def pack(self):
        initiateParams = self.outputShape,self.activation,self.inputShape
        w,b = self.w.get(), self.b.get()
        return (self.__class__, initiateParams, w, b)

        

class Conv(Layer):
    def __init__(self, kernel, filters, padding, strides, activation, inputShape =  None):
        self.kernel = np.int32(kernel)
        self.filters = np.int32(filters)
        self.padding = 0#np.int32(padding)
        self.activation = activation
        self.strides = tuple(np.int32(i) for i in strides)
        if inputShape is not None:
            self.initiateInput(inputShape)
        else:
            self.inputShape = None
        
    def initiateInput(self, inputShape):
        self.inputShape = tuple(np.int32(i) for i in inputShape)
        padding = self.padding
        strides = self.strides    
        kernel = self.kernel
        filters = self.filters
        self.outputShape = (*tuple((size+2*padding-kernel[dim])//strides[dim] + 1
                        for dim,size in enumerate(inputShape[:2])),filters)
        self.w = Tensor(np.random.randn(*(filters, *kernel, inputShape[2])))
        self.b = Tensor(np.random.randn(filters))

    def allocateMemory(self, batchSize):
        self.batchSize = np.int32(batchSize)
        self.batchSize = batchSize
        self.v = Tensor((self.batchSize,*self.outputShape))
        self.y = Tensor((self.batchSize,*self.outputShape))
        self.dphi = Tensor((self.batchSize,*self.outputShape))
        self.sigma = Tensor((self.batchSize,*self.inputShape))
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
        self.computedb = Kernel(programConv.computedb, 
             (batchSize,self.filters),
             (*self.outputShape,self.dphi,self.db))
        self.computeGradients = Kernel(programConv.computeGradients, 
             (batchSize*self.filters,np.prod(self.kernel),self.inputShape[2]),
             (*self.outputShape,*self.inputShape,*self.kernel, *self.strides,
             self.dw,self.dphi))
        self.computeLocalGradient = Kernel(programConv.computeLocalGradient, 
            (batchSize, np.prod(self.inputShape[:2]), self.inputShape[2]),
            (*self.outputShape,*self.inputShape,*self.kernel, *self.strides,
            self.sigma, self.dphi, self.w))
        self.learningRule = Kernel(programConv.learningRule, 
            (np.prod(self.w.shape),),
            (self.filters, self.batchSize, np.prod(self.w.shape[1:]), self.dw,self.db,self.w,self.b))

    def forwardPropagate(self, ym1):
        self.ym1 = ym1
        self.GPUForwardPropagate(self.ym1)
        self.activate()
        return self.y
    
    def backwardPropagate(self, sigma, lrate):
        self.computedb(sigma)
        self.computeGradients(self.ym1, sigma)
        self.computeLocalGradient(sigma)
        self.learningRule(np.float32(lrate))
        return self.sigma
    
    def pack(self):
        initiateParams = self.kernel, self.filters, self.padding, \
                self.strides, self.activation, self.inputShape
        w,b = self.w.get(), self.b.get()
        return (self.__class__, initiateParams, w, b)

    def unpack(initiateParams,w,b):
        kernel, filters, padding, strides, activation, inputShape = initiateParams
        instance = Conv(kernel, filters, padding, strides, activation, inputShape= inputShape)
        instance.w.set(w)
        instance.b.set(b)

        
class Reshape(Layer):
    PARAMETERS = False
    def __init__(self, outputShape, inputShape = None):
        self.w = None
        self.b = None
        if isinstance(outputShape,int):
            self.outputShape = (np.int32(outputShape),)
        else:
            self.outputShape = outputShape
        if inputShape is not None:
            self.initiateInput(inputShape)
        else:
            self.inputShape = None
        
    def initiateInput(self, inputShape):
        self.outputShape = Tensor.getSpecifiedShape(inputShape,self.outputShape)
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
    
    def unpack(initiateParams,w,b):
        outputShape, inputShape = initiateParams
        instance = Reshape(outputShape, inputShape= inputShape)
        
    def pack(self):
        initiateParams = self.outputShape, self.inputShape
        return (self.__class__, initiateParams)