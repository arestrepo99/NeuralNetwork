from xml.dom.expatbuilder import parseString
import numpy as np
import pyopencl as cl
from Tensor import Tensor
from functools import reduce
from settings import queue, kerneloptimization
from time import time
from itertools import product

def factors(n):    
    return set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))) | {n}

class Kernel():
    def __init__(self, function, globalSize, tensors = tuple(), constants= tuple()):
        self.function = function
        self.tensors  = tensors# tuple(Kernel.unpack(arg) for arg in tensors)
        self.constants =  constants #tuple(Kernel.unpack(arg) for arg in constants)
        self.globalSize = globalSize
        self.localSize = None
        if not isinstance(self.globalSize,tuple):
            raise TypeError("Global Size must be a tuple")

    def unpack(arg):
        if isinstance(arg,int) or isinstance(arg, np.int64):
            return np.int32(arg)
        if isinstance(arg,float) or isinstance(arg, np.float64):
            return np.float32(arg)
        if isinstance(arg,Tensor):
            return arg.buffer
        raise Exception(f"Kernel argument with type {type(arg)} has incorrect type, only int, float, Tensor allowed")

    def __call__(self, *args):
        if self.localSize is None:
            self.optimize(args)
        unpacked_args = tuple(Kernel.unpack(arg) for arg in args+self.constants+self.tensors)
        self.function(queue, self.globalSize, self.localSize, *unpacked_args)


    def optimize(self, args, reps = 3):   
        if kerneloptimization:
            run_time = {}
            for size in product(*[factors(size) for size in self.globalSize]):
                self.localSize = size
                try:
                    queue.finish()
                    start_time = time()
                    for _ in range(reps):
                        self(*args)
                        queue.finish()
                    run_time[size] = time()-start_time
                except cl.LogicError as e:
                    continue
            self.localSize = min(run_time, key=run_time.get)
            self.time = run_time[self.localSize]
        else:
            self.localSize = None