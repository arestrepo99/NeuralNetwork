import pyopencl as cl

context = cl.Context()
queue = cl.CommandQueue(context)
program = cl.Program(context, open('kernels.cl').read()).build()
programConv = cl.Program(context, open('kernelsConv.cl').read()).build()