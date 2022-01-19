import numpy as np
import pyopencl as cl
import time

context = cl.Context()
queue = cl.CommandQueue(context)
program = cl.Program(context, open('kernels.cl').read()).build()
programConv = cl.Program(context, open('kernelsConv.cl').read()).build()


class Tensor:
    def __init__(self, input, shape = None):
        if isinstance(input, cl.Buffer):
            self.buffer = input
            self.shape = shape
        elif isinstance(input, np.ndarray):
            self.shape = input.shape
            self.buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE,
                        size=np.prod(self.shape)*32//4)
            self.set(input)
        elif isinstance(input, tuple):
            self.shape = input
            self.buffer = cl.Buffer(context,cl.mem_flags.READ_WRITE,
                        size=np.prod(self.shape)*32//4)
        else:
            raise(TypeError("Expected buffer, numpy array, or shape"))
    
    def get(self):
        out = np.empty(self.shape).astype(np.float32)
        cl.enqueue_copy(queue, src=self.buffer, dest=out)
        return out

    def __repr__(self) -> str:
        return self.get().__str__()

    def set(self,values):
        cl.enqueue_copy(queue, src=values.flatten().astype(np.float32), 
                    dest=self.buffer)

    def __getitem__(self, key):
        size = np.prod(self.shape[1:])
        return Tensor(self.buffer[key*size*4:(key+1)*size*4], shape = self.shape[1:])
    
    
class Kernel():
    def __init__(self, function, globalSize, staticParams):
        self.function = function
        self.staticParams  = staticParams
        self.globalSize = globalSize
        self.localSize = None
    

    def __call__(self, *params):
        def getBuffers(params):
            out = []
            for param in params:
                if isinstance(param,Tensor):
                    out.append(param.buffer)
                elif isinstance(param,int):
                    out.append(np.int32(param))
                else:
                    out.append(param)
            return tuple(out)
        if self.localSize is None:
            self.optimize(params)
        self.function(queue, self.globalSize, self.localSize, 
                *getBuffers(params), *getBuffers(self.staticParams))

    def optimize(self, params, reps = 2, debug = True):        
        self.localSize = tuple([1]*len(self.globalSize))

class Layer:
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

class Dense(Layer):
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
        

class Conv(Layer):
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

    def backwardPropagate(self, **kwargs):
        if 'Y' in kwargs:
            self.computeError(kwargs['Y'], kwargs['e'])
            self.computedb(kwargs['e'])
            self.computeGradients(self.ym1, kwargs['e'])
        #if 'sigma' in kwargs:
        #    self.computedb(kwargs['sigma'])
        #    self.computeGradients(self.ym1, kwargs['sigma'])
        #self.computeLocalGradient()
        #self.learningRule(np.float32(kwargs['lrate']))
        return self.sigma

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
        

    



class NeuralNetowrk:
    def __init__(self, layers):
        self.layers = layers
        self.inputShape = layers[0].inputShape
        self.outputShape = layers[-1].outputShape
        self.loss = []

    def getLoss(self,):
            self.squareError()
            self.meanError()
            return(self.E.get().mean())

    def allocateMemory(self, batchSize):
        self.batchSize = np.int32(batchSize)
        
        self.e = Tensor((self.batchSize,self.outputShape))
        self.e2 = Tensor((self.batchSize,self.outputShape))
        self.E = Tensor((self.outputShape,))

        [layer.allocateMemory(batchSize) for layer in self.layers]
        self.squareError = Kernel(program.squareError, (self.batchSize,self.outputShape),
                        (self.e, self.e2, self.outputShape))
        
        self.meanError = Kernel(program.meanError, (self.batchSize,self.outputShape),
                        (self.outputShape, self.batchSize, self.e2, self.E))
        
    
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
            w = layer.w.get().flatten()
            for i in range(w.size):
                w0 = w[i]
                w[i] = w0 + step
                layer.w.set(w)
                self.gradientDescent(X,Y,lrate = 0)
                dw[-1].append((self.getLoss()-l)/step)
                w[i] = w0
                layer.w.set(w)
            db.append([])
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
import matplotlib.pyplot as plt
sigmoid = lambda x: x.sigmoid

#https://learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/