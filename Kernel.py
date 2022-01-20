import numpy as np
import pyopencl as cl
from Tensor import Tensor

from settings import queue
 
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