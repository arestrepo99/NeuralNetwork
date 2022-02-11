import pyopencl as cl

context = cl.Context()
queue = cl.CommandQueue(context)
densecl = cl.Program(context, open('Kernels/Dense.cl').read()).build()
convolutionalcl = cl.Program(context, open('Kernels/Convolutional.cl').read()).build()
activationscl = cl.Program(context, open('Kernels/Activations.cl').read()).build()
neuralnetworkcl = cl.Program(context, open('Kernels/NeuralNetwork.cl').read()).build()
#poolcl = cl.Program(context, open('Pool.cl').read()).build()

kerneloptimization = False

run_times = []