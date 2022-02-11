from matplotlib.container import BarContainer
import numpy as np
from utils.Tensor import Tensor
from utils.Kernel import Kernel
from settings import densecl, convolutionalcl, activationscl

class sigmoid:
    kernel = activationscl.sigmoid


class Layer:
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'{self.__class__.__name__} Layer: \t\t{self.outputShape}\n'

class Dense(Layer):
    def __init__(self, outputShape, activation, inputShape = None ):
        if isinstance(outputShape,int):
            self.outputShape = (outputShape,)
        else:
            self.outputShape = outputShape
        self.activation = activation
        if inputShape is not None:
            self.initiateInput(inputShape)
        else:
            self.inputShape = None
        
    def initiateInput(self, inputShape):
        self.inputShape = inputShape
        self.w = Tensor(np.random.randn(*inputShape,*self.outputShape))
        self.b = Tensor(np.random.randn(*self.outputShape))

    def allocateMemory(self, batchSize):
        outputSize, = self.outputShape
        inputSize, = self.inputShape
        batchSize = batchSize
        self.batchSize = batchSize
        self.v = Tensor((batchSize,outputSize))
        self.y = Tensor((batchSize,outputSize))
        self.dphi = Tensor((batchSize,outputSize))
        self.sigma = Tensor((batchSize,inputSize))
        self.dw = Tensor((batchSize,inputSize,outputSize))
        self.db = Tensor((batchSize,outputSize))

        Kernel(densecl.forwardPropagate, globalSize= tuple(), tensors= tuple(), constants= tuple())
        self.GPUForwardPropagate = Kernel(densecl.forwardPropagate, 
            globalSize= (batchSize,outputSize), 
            tensors= (self.v, self.w, self.b), 
            constants= (inputSize, outputSize) )
        self.activate = Kernel(self.activation.kernel, 
            globalSize= (batchSize*outputSize,),
            tensors = (self.v, self.y, self.dphi),
            constants = tuple() )
        self.computedb = Kernel(densecl.computedb, 
            globalSize= (batchSize*outputSize,),
            tensors= (self.dphi, self.db),
            constants= tuple())
        self.computeGradients = Kernel(densecl.computeGradients, 
            globalSize= (batchSize, inputSize, outputSize),
            tensors= (self.dw, self.db),
            constants= (inputSize, outputSize)    )
        self.computeLocalGradient = Kernel(densecl.computeLocalGradient, 
            globalSize= (batchSize, inputSize),
            tensors= (self.sigma, self.db, self.dphi, self.w),
            constants= (inputSize, outputSize,))
        self.learningRule = Kernel(densecl.learningRule, 
            globalSize= (inputSize, outputSize),
            tensors= (self.dw,self.db,self.w,self.b),
            constants= (inputSize, outputSize, batchSize))
    

    def forwardPropagate(self, ym1):
        self.ym1 = ym1
        self.GPUForwardPropagate(self.ym1)
        self.activate()
        return self.y
    
    def backwardPropagate(self, lrate, sigmaOut):
        self.sigmaOut = sigmaOut
        self.computedb(self.sigmaOut)
        self.computeGradients(self.ym1)
        self.computeLocalGradient()
        self.learningRule(lrate)
        return self.sigma
    
    def unpack(params):
        outputShape, activation, inputShape, w, b = params
        instance = Dense(outputShape, activation, inputShape= inputShape)
        instance.w.set(w)
        instance.b.set(b)
        return instance

    def pack(self):
        initiateParams = self.outputShape,self.activation,self.inputShape, self.w.get(), self.b.get()
        return (self.__class__, initiateParams)


class Conv(Layer):
    def __init__(self, kernel, activation, filters = 1, padding= 0, strides= (1,1), inputShape =  None):
        self.kernel = kernel
        self.filters = filters
        self.padding = padding
        self.activation = activation
        self.strides = strides
        if inputShape is not None:
            self.initiateInput(inputShape)
        else:
            self.inputShape = None
        
    def initiateInput(self, inputShape):
        self.inputShape = inputShape
        padding = self.padding
        strides = self.strides    
        kernel = self.kernel
        filters = self.filters
        self.outputShape = (*tuple((size+2*padding-kernel[dim])//strides[dim] + 1
                        for dim,size in enumerate(inputShape[:2])),filters)
        self.w = Tensor(np.random.randn(*(filters, *kernel, inputShape[2])))
        self.b = Tensor(np.random.randn(filters))

    def allocateMemory(self, batchSize):
        self.batchSize = batchSize
        self.batchSize = batchSize
        self.v = Tensor((self.batchSize,*self.outputShape))
        self.y = Tensor((self.batchSize,*self.outputShape))
        self.dphi = Tensor((self.batchSize,*self.outputShape))
        self.sigma = Tensor((self.batchSize,*self.inputShape))
        self.dw = Tensor((batchSize, self.filters, *self.kernel, self.inputShape[2]))
        self.db = Tensor((batchSize, self.filters))
        
        self.GPUForwardPropagate = Kernel(convolutionalcl.forwardPropagate,
            globalSize= (batchSize, self.filters, np.prod(self.outputShape[:2])),
            tensors= (self.v, self.w, self.b),
            constants= (*self.outputShape[:2], self.filters,*self.strides, *self.kernel,
                                                         *self.inputShape, self.padding))

        self.activate = Kernel(self.activation.kernel, 
            globalSize= (np.prod((batchSize,*self.outputShape)),) ,
            tensors= (self.v, self.y, self.dphi),
            constants= tuple())

        self.computedb = Kernel(convolutionalcl.computedb, 
            globalSize= (batchSize,self.filters),
            tensors= (self.dphi,self.db),
            constants= self.outputShape)

        self.computeGradients = Kernel(convolutionalcl.computeGradients, 
            globalSize= (batchSize*self.filters,np.prod(self.kernel),self.inputShape[2]),
            tensors= (self.dw,self.dphi),
            constants= (*self.outputShape,*self.inputShape,*self.kernel, *self.strides, self.padding))

        self.computeLocalGradient = Kernel(convolutionalcl.computeLocalGradient, 
            globalSize= (batchSize, np.prod(self.inputShape[:2]), self.inputShape[2]),
            tensors= (self.sigma, self.dphi, self.w),
            constants= (*self.outputShape,*self.inputShape,*self.kernel, *self.strides, self.padding) )

        self.learningRule = Kernel(convolutionalcl.learningRule, 
            globalSize= (np.prod(self.w.shape),),
            tensors= (self.dw,self.db,self.w,self.b),
            constants= (self.filters, self.batchSize, np.prod(self.w.shape[1:])) )


    def forwardPropagate(self, ym1):
        #Conv
        self.ym1 = ym1
        self.GPUForwardPropagate(ym1)
        self.activate()
        return self.y
    
    def backwardPropagate(self, lrate, sigmaOut):
        #Conv
        self.sigmaOut = sigmaOut 
        self.computedb(sigmaOut)
        self.computeGradients(self.ym1, sigmaOut)
        self.computeLocalGradient(self.sigmaOut)
        self.learningRule(lrate)
        return self.sigma
    
    def pack(self):
        initiateParams = self.kernel, self.filters, self.padding, \
                self.strides, self.activation, self.inputShape, self.w.get(), self.b.get()
        w,b = self.w.get(), self.b.get()
        return (self.__class__, initiateParams)

    def unpack(params):
        kernel, filters, padding, strides, activation, inputShape, w, b = params
        instance = Conv(kernel, activation, filters=filters, padding=padding,
                                         strides=strides, inputShape= inputShape)
        instance.w.set(w)
        instance.b.set(b)
        return instance

        
class Reshape(Layer):
    PARAMETERS = False
    def __init__(self, outputShape, inputShape = None):
        self.w = None
        self.b = None
        if isinstance(outputShape,int):
            self.outputShape = (outputShape,)
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
    
    def backwardPropagate(self, lrate, sigmaOut):
        self.sigmaOut = sigmaOut
        self.sigma = self.sigmaOut.reshape((self.batchSize,*self.inputShape))
        return self.sigma
    
    def unpack(initiateParams):
        outputShape, inputShape = initiateParams
        instance = Reshape(outputShape, inputShape= inputShape)
        return instance
        
    def pack(self):
        initiateParams = self.outputShape, self.inputShape
        return (self.__class__, initiateParams)

#https://learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/