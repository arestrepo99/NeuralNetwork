# Testing kernels
from ast import Slice
import numpy as np
from NeuralNetwork import NeuralNetwork
from Layers import Dense, Reshape, Conv, sigmoid
from Tensor import Tensor
from Kernel import Kernel
from itertools import product
from inspect import signature
from sklearn.linear_model import LinearRegression

from testsConv import forwardPropagate

TOLERANCE = 1e-4

def parse(kernelFuncion):
    kernelFuncion = kernelFuncion.replace('kernel void','def')
    kernelFuncion = kernelFuncion.replace('global ','')
    kernelFuncion = kernelFuncion.replace('float','')
    kernelFuncion = kernelFuncion.replace('const','')
    kernelFuncion = kernelFuncion.replace('uint','')
    kernelFuncion = kernelFuncion.replace('get_global_id','globalIndex')
    kernelFuncion = kernelFuncion.replace('{',':')
    kernelFuncion = kernelFuncion.replace('}','')
    kernelFuncion = kernelFuncion.replace(';','')
    return kernelFuncion

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    Passed = OKGREEN + "PASSED" + ENDC
    Failed = FAIL + "FAILED" + ENDC

class TestArray():
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.accessed = np.zeros(array.shape).astype(bool) 

    def __getitem__(self, index: tuple):
        for i,ind in enumerate(index):
            if not (ind >= 0):
                print(index)
            assert ind >= 0, f'dim {i}'
            assert ind < self.array.shape[i], f'dim {i}'
        self.accessed[index] = True
        return self.array[index]
    
    def __setitem__(self, index: tuple, newvalue):
        for i,ind in enumerate(index):
            assert ind >= 0, f'dim {i}'
            assert ind < self.array.shape[i], f'dim {i}'
        self.accessed[index] = True
        self.array[index] = newvalue
    
    def assertAccessed(self,paramname):
        if not self.accessed.all():
            print(f'Warning: Not all of {paramname} accesed')
            return True
        return False


def test(kernel: Kernel, function, args): 
    params = []
    for param in list(args)+list(kernel.staticParams):
        if isinstance(param,Tensor):
            params.append(TestArray(param.get()))
        else:
            params.append(param)
    # Running in CPU
    for globalIndex in product(*(list(range(i)) for i in kernel.globalSize)):
        output = function(globalIndex,*params)
    # Running in GPU
    output = kernel(*args)

    #Getting Param Names 
    paramNames = str(signature(function))[1:-1].split(', ')[1:]

    #Compare outputs to determine if test was passed or failed
    passed = True
    message = ""
    debugOutput = {}
    for ind,param in enumerate(args+kernel.staticParams):
        if isinstance(param,Tensor):
            param = param.get()
            if params[ind].assertAccessed(paramNames[ind]):
                debugOutput["na"+paramNames[ind]] = params[ind].accessed
            if abs(np.sum(params[ind].array-param)) > TOLERANCE:
                message += f'Param Index {ind} with name "{paramNames[ind]}" was off by {abs(np.sum(params[ind].array-param))}\n'
                passed = False
                debugOutput["d"+paramNames[ind]] = params[ind].array-param
    if passed:
        print(f'{bcolors.Passed} {function.__name__}')
    else:
        print(f'{bcolors.Failed} {function.__name__}')
        print(message)
    return output,debugOutput


def getRandomData(model: NeuralNetwork, batchSize):
    model.allocateMemory(batchSize)
    X = Tensor(np.random.randn(batchSize,*model.inputShape))
    Y = Tensor(np.random.randn(batchSize,*model.outputShape))
    return X,Y


def gradientTest(model: NeuralNetwork, step = 0.1):
    X,Y = getRandomData(model,1)
    model.gradientDescent(X, Y, 0)
    dw = []
    db = []
    l = model.loss.pop()
    for ind,layer in enumerate(model.layers):
        #Approximating Gradients Numerically
        dw.append([])
        db.append([])
        if isinstance(layer,Reshape):
            continue
        w = layer.w.get().flatten()
        for i in range(w.size):
            w0 = w[i]
            w[i] = w0 + step
            layer.w.set(w)
            model.predict(X)
            dw[-1].append((model.getLoss(Y)-l)/step)
            w[i] = w0
            layer.w.set(w)
        b = layer.b.get().flatten()
        for i in range(b.size):
            b0 = b[i]
            b[i] = b0 + step
            layer.b.set(b)
            model.predict(X)
            db[-1].append((model.getLoss(Y)-l)/step)
            b[i] = b0
            layer.b.set(b)
        # Checking If Aproximated gradients match anatlitical gradients
        PRESITION = 2
        params = [(np.array(dw[-1]).reshape(-1,1),layer.dw.get().flatten()),
                 (np.array(db[-1]).reshape(-1,1),layer.db.get().flatten())]
        for paramInd,param in enumerate(params):
            x,y = param
            reg = LinearRegression().fit(x,y)
            m = round(reg.coef_[0],PRESITION)
            b = round(reg.intercept_,PRESITION)
            r2 = round(reg.score(x,y),PRESITION)
            paramName = ['dw','db'][paramInd]
            if abs(r2-1) < TOLERANCE:
                message = bcolors.Passed
            else:
                message = bcolors.Failed
            print(f'{message} Layer {ind} {paramName} r2= {r2}, m= {m}, b= {b},')
    def plotter(ind):
        import matplotlib.pyplot as plt
        plt.scatter(model.layers[ind].dw.get().flatten(),dw[ind])
        plt.scatter(model.layers[ind].db.get().flatten(),db[ind])
    
    return dw,db,plotter
        
def testModel(model):
    dw,db = gradientTest(model)
    X,Y = getRandomData(model,10)
    ym1 = X
    for layer in model.layers:
        ym1 = test(layer.forwardPropagate(ym1), forwardPropagate, ym1)
    model.ypred = ym1
    test(model.computeError,computeError,)

