# Testing kernels
import numpy as np
from NeuralNetwork import NeuralNetwork, sigmoid
from Dense import Dense, Reshape
from Conv import Conv
from Tensor import Tensor
from Kernel import Kernel
from itertools import product
from inspect import signature


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

def test(kernel: Kernel,function,args): 
    params = []
    for param in list(args)+list(kernel.staticParams):
        if isinstance(param,Tensor):
            params.append(param.get())
        else:
            params.append(param)
    # Running in CPU
    for globalIndex in product(*(list(range(i)) for i in kernel.globalSize)):
        function(globalIndex,*params)
    # Running in GPU
    kernel(*args)

    #Getting Param Names 
    paramNames = str(signature(function))[1:-1].split(', ')[1:]

    #Compare outputs to determine if test was passed or failed
    passed = True
    message = ""
    differingParams = {}
    for ind,param in enumerate(args+kernel.staticParams):
        if isinstance(param,Tensor):
            param = param.get()
        if abs(np.sum(params[ind]-param)) > 1e-5:
            message += f'Param Index {ind} with name "{paramNames[ind]}" was off by {abs(np.sum(params[ind]-param))}\n'
            passed = False
            differingParams[paramNames[ind]] = (params[ind],param)
    if passed:
        print(bcolors.OKGREEN + "Test Passed" + bcolors.ENDC, function.__name__, )
    else:
        print(bcolors.FAIL + "Test Failed" + bcolors.ENDC, function.__name__)
        print(message)
    return differingParams


