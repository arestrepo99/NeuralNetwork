from xml.dom.expatbuilder import parseString
import numpy as np
import pyopencl as cl
from utils.Tensor import Tensor
from functools import reduce
from settings import queue, kerneloptimization
import settings
from time import time
from itertools import product

def factors(n):    
    return set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))) | {n}

class Kernel():
    run_times = []
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
        unpacked_args = tuple(Kernel.unpack(arg) for arg in args+self.tensors+self.constants)
        queue.finish()
        start_time = time()
        #print(unpacked_args)
        try:
            self.function(queue, self.globalSize, self.localSize, *unpacked_args)
        except:
            print(*unpacked_args)
            print(self.function.function_name)
            raise Exception
        queue.finish()
        settings.run_times.append((self.function.function_name,time()-start_time))
    def optimize(self, args, reps = 10):   
        if kerneloptimization:
            run_time = {}
            for size in product(*[factors(size) for size in self.globalSize]):
                try:   
                    unpacked_args = tuple(Kernel.unpack(arg) for arg in args+self.tensors+self.constants)
                    queue.finish()
                    start_time = time()
                    for _ in range(reps):
                        self.function(queue, self.globalSize, size, *unpacked_args)
                    queue.finish()
                    run_time[size] = time()-start_time
                    
                except cl.LogicError as e:
                    continue
            self.localSize = min(run_time, key=run_time.get)
            self.time = run_time[self.localSize]
            print(f"Optimized {self.function.function_name}")
        else:
            self.localSize = None