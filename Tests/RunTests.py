from NeuralNetwork import NeuralNetwork
from Layers import *
from Tests.AproximateGradientTest import gradientTest
model = NeuralNetwork([Conv((5,5),sigmoid, filters = 3, padding = 2, strides=(2,2),inputShape=(10,10,1)),
                        #Conv((3,3),6,0,(1,1),sigmoid),
                        Conv((2,2),sigmoid,filters = 6), # GRADIENT LOCALS NOT WORKING PROPPERLY
                        Reshape(-1), 
                        Dense(3,sigmoid),
                        Reshape(-1), 
                        Dense(3,sigmoid)])
                        
dw,db,plotter = gradientTest(model)


from NeuralNetwork import NeuralNetwork
from Layers import Dense, Conv, Reshape, sigmoid
from Tests import CPUvsGPU

conv1 = model.layers[0]
conv1.allocateMemory(10)


CPUvsGPU.testConv(conv1)

