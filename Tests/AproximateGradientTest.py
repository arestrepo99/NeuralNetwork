from NeuralNetwork import NeuralNetwork
from Tensor import Tensor
import numpy as np
from Layers import Reshape
from sklearn.linear_model import LinearRegression

def getRandomData(model: NeuralNetwork, batchSize):
    model.allocateMemory(batchSize)
    X = Tensor(np.random.randn(batchSize,*model.inputShape))
    Y = Tensor(np.random.randn(batchSize,*model.outputShape))
    return X,Y

OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
UNDERLINE = '\033[4m'
Passed = OKGREEN + "PASSED" + ENDC
Failed = FAIL + "FAILED" + ENDC

TOLERANCE = 5e-2


def gradientTest(model: NeuralNetwork, step = 0.1):
    X,Y = getRandomData(model,1)
    model.gradientDescent(X, Y, 0)
    dw = []
    db = []
    l = model.loss.pop()
    for ind,layer in enumerate(model.layers):
        #Approximating Gradients Numerically
        dw.append([])
        db.append([])
        if isinstance(layer,Reshape):
            continue
        w = layer.w.get().flatten()
        for i in range(w.size):
            w0 = w[i]
            w[i] = w0 + step
            layer.w.set(w)
            model.predict(X)
            dw[-1].append((model.getLoss(Y)-l)/step)
            w[i] = w0
            layer.w.set(w)
        b = layer.b.get().flatten()
        for i in range(b.size):
            b0 = b[i]
            b[i] = b0 + step
            layer.b.set(b)
            model.predict(X)
            db[-1].append((model.getLoss(Y)-l)/step)
            b[i] = b0
            layer.b.set(b)
        # Checking If Aproximated gradients match anatlitical gradients
        PRESITION = 2
        params = [(np.array(dw[-1]).reshape(-1,1),layer.dw.get().flatten()),
                 (np.array(db[-1]).reshape(-1,1),layer.db.get().flatten())]
        for paramInd,param in enumerate(params):
            x,y = param
            reg = LinearRegression().fit(x,y)
            m = round(reg.coef_[0],PRESITION)
            b = round(reg.intercept_,PRESITION)
            r2 = round(reg.score(x,y),PRESITION)
            paramName = ['dw','db'][paramInd]
            if abs(r2-1) < TOLERANCE:
                message = Passed
            else:
                message = Failed
            print(f'{message} Layer {ind} {paramName} r2= {r2}, m= {m}, b= {b},')
    def plotter(ind):
        import matplotlib.pyplot as plt
        plt.scatter(model.layers[ind].dw.get().flatten(),dw[ind])
        plt.scatter(model.layers[ind].db.get().flatten(),db[ind])
    
    return dw,db,plotter
