import numpy as np
import pyopencl as cl
from Tensor import Tensor
from functools import reduce
from settings import queue, kerneloptimization
from time import time
from itertools import product

class InvalidGlobalSize(Exception):
    pass
class InvalidParameters(Exception):
    pass

def factors(n):    
    return set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))) | {n}

class Kernel():
    def __init__(self, function, globalSize, staticParams):
        self.function = function
        self.staticParams  = staticParams
        self.globalSize = globalSize
        self.localSize = None
        if not isinstance(self.staticParams,tuple):
            raise InvalidParameters("Parameters must be a tuple")
        if not isinstance(self.staticParams,tuple):
            raise InvalidGlobalSize("Global Size must be a tuple")

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


    def optimize(self, params, reps = 3):   
        if kerneloptimization:
            run_time = {}
            for size in product(*[factors(size) for size in self.globalSize]):
                self.localSize = size
                try:
                    start_time = time()
                    for _ in range(reps):
                        self(*params)
                        queue.finish()
                    run_time[size] = time()-start_time
                except cl.LogicError as e:
                    continue
            self.localSize = min(run_time, key=run_time.get)
            self.time = run_time[self.localSize]
        else:
            self.localSize = tuple([1]*len(self.globalSize))