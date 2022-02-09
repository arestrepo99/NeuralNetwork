import pyopencl as cl

context = cl.Context()
queue = cl.CommandQueue(context)
densecl = cl.Program(context, open('Dense.cl').read()).build()
convolutionalcl = cl.Program(context, open('Convolutional.cl').read()).build()
activationscl = cl.Program(context, open('Activations.cl').read()).build()
#poolcl = cl.Program(context, open('Pool.cl').read()).build()

kerneloptimization = True