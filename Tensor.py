import numpy as np
import pyopencl as cl

from settings import context, queue

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
    
    def getSpecifiedShape(shape,reshape):
        for ind,size in enumerate(reshape):
            if size == -1:
                reshape = list(reshape)
                reshape[ind] = -np.prod(shape)//np.prod(reshape)
                if -np.prod(shape)%np.prod(reshape) != 0:
                    raise Exception("Axis not reshapable")
                reshape = tuple(reshape)
        return reshape

    def reshape(self, shape):
        shape = Tensor.getSpecifiedShape(self.shape,shape)
        if not np.prod(shape) == np.prod(self.shape):
            raise Exception(f'New shape length {np.prod(shape)} does not match ' + \
                f'old shape length {np.prod(self.shape)} dimension')
        return Tensor(self.buffer, shape = shape)