
# Testing kernels
from ast import Slice
import numpy as np
#from NeuralNetwork import NeuralNetwork
#from Layers import Dense, Reshape, Conv, sigmoid
from Tensor import Tensor
from Kernel import Kernel
from itertools import product
from inspect import signature


OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
UNDERLINE = '\033[4m'
Passed = OKGREEN + "PASSED" + ENDC
Failed = FAIL + "FAILED" + ENDC

TOLERANCE = 1e-4

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
            print(f'{WARNING}Warning:{ENDC} Not all of {paramname} accesed')
            return True
        return False

def runCPUKernel(function, globalSize, args):
    testArrayParameters = {}
    #Getting Param Names 
    PARAMETER_NAMES = str(signature(function))[1:-1].split(', ')[1:]
    # Changing tensors to test array
    for ind,param in enumerate(args):
        if isinstance(param,Tensor):
            testArrayParameters[PARAMETER_NAMES[ind]] = TestArray(param.get())
        else:
            testArrayParameters[PARAMETER_NAMES[ind]] = param
    # Running in CPU
    for globalIndex in product(*(list(range(i)) for i in globalSize)):
        function(globalIndex,*testArrayParameters.values())
    return testArrayParameters

def test(kernel: Kernel, function, args):
    # Running CPU Kernel
    
    CPU_PARAMS = runCPUKernel(function, kernel.globalSize, args+kernel.tensors+kernel.constants)
    # Running GPU Kernel
    kernel(*args)

    #Getting Param Names 
    PARAMETER_NAMES = str(signature(function))[1:-1].split(', ')[1:]
    GPU_PARAMS = {}
    for ind,param in enumerate(args+kernel.tensors+kernel.constants):
        GPU_PARAMS[PARAMETER_NAMES[ind]] = param
    #Compare outputs to determine if test was passed or failed
    passed = True
    message = ""
    for name in PARAMETER_NAMES:
        if isinstance(param,Tensor):
            mean_error = np.sum(np.abs(CPU_PARAMS[name].array-GPU_PARAMS[name].get()))
            if mean_error > TOLERANCE:
                message += f'Param Index {ind} with name "{name}" was off by {mean_error}\n'
                passed = False
    if passed:
        print(f'{Passed} {function.__name__}')
    else:
        print(f'{Failed} {function.__name__}')
        print(message)
    #Asserting if Accessed
    for name,param in CPU_PARAMS.items():
        if isinstance(param,TestArray):
            param.assertAccessed(name)


import Tests.Convolutional as Convolutional
def testConv(conv):
    ym1 = Tensor(np.random.randn(conv.batchSize,*conv.inputShape))
    sigmaOut = Tensor(np.random.randn(conv.batchSize,*conv.outputShape))
    test(conv.GPUForwardPropagate,Convolutional.forwardPropagate,(ym1,))
    test(conv.activate,Convolutional.sigmoidTest,tuple())
    test(conv.computedb,Convolutional.computedb,(sigmaOut,))
    test(conv.computeGradients,Convolutional.computeGradients,(ym1,sigmaOut))
    test(conv.computeLocalGradient,Convolutional.computeLocalGradient,(sigmaOut,))
    test(conv.learningRule,Convolutional.learningRule,(0.1,))