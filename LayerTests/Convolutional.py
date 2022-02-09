from sre_constants import IN_IGNORE
import numpy as np
def forwardPropagate(globalIndex,
                    ym1,
                    outSize1,
                    outSize2,
                    filters,
                    stride1,
                    stride2,
                    kernel1,
                    kernel2,
                    inSize1,
                    inSize2,
                    inSize3,
                    padding,
                    v,
                    w,
                    b):

    batch = globalIndex[0]
    filter = globalIndex[1]
    out1 = globalIndex[2]%outSize1
    out2 = globalIndex[2]//outSize1
    
    in1 = out1*stride1-padding
    in2 = out2*stride2-padding

    K1_LIMIT_INF = max(0,-in1)
    K2_LIMIT_INF = max(0,-in2)
    K1_LIMIT_SUP = min(kernel1,inSize1-in1)
    K2_LIMIT_SUP = min(kernel2,inSize2-in2)

    v[batch,out1,out2,filter] = b[filter,]
    for k1 in range(K1_LIMIT_INF, K1_LIMIT_SUP):
        for k2 in range(K2_LIMIT_INF, K2_LIMIT_SUP):
            for dim in range(inSize3):
                in1 = out1*stride1-padding+k1
                in2 = out2*stride2-padding+k2
                v[batch,out1,out2,filter] += \
                ym1[batch,in1,in2,dim] * \
                w[filter, k1, k2,dim]

#k1 goes form  max(0,-in1) to  min(kernel1,inSize1-in1)

#input1 = output1*stride1+k1-padding
#output1 = (input1+padding-k1)/stride1

def computedb(globalIndex,
             sigmaOut,
             outSize1,
             outSize2,
             filters,
             dphi,
             db):

    batch = globalIndex[0]
    filter = globalIndex[1]

    ind = batch*filters+filter
    indOut = batch*filters*outSize1*outSize2+filter

    db[np.unravel_index(ind,db.shape)] = 0
    for o1 in range(outSize1):
        for o2 in range(outSize2):
            indOut2 = o1*outSize2*filters +  o2*filters
            db[np.unravel_index(ind,db.shape)] += \
                sigmaOut[np.unravel_index(indOut+indOut2,sigmaOut.shape)] \
                *dphi[np.unravel_index(indOut+indOut2,dphi.shape)]

def computeGradients(globalIndex,
                        ym1,
                        sigmaOut,
                        outSize1,
                        outSize2,
                        filters,
                        inSize1,
                        inSize2,
                        inSize3,
                        kernel1,
                        kernel2,
                        stride1,
                        stride2,
                        padding,
                        dw,
                        dphi
                        ):
    batch = globalIndex[0]//filters
    filter = globalIndex[0]%filters
    k1 = globalIndex[1]//kernel2
    k2 = globalIndex[1]%kernel2
    dim = globalIndex[2]

    IN1_LIMIT_INF = max(0,k1-padding) +(stride1-max(0,-k1+padding))%stride1
    IN2_LIMIT_INF = max(0,k2-padding) +(stride2-max(0,-k2+padding))%stride2
    IN1_LIMIT_SUP = min(inSize1,outSize1*stride1-padding+k1)
    IN2_LIMIT_SUP = min(inSize2,outSize2*stride2-padding+k2)
    dw[batch,filter,k1,k2,dim] = 0
    for in1 in range(IN1_LIMIT_INF,IN1_LIMIT_SUP,stride1):
        for in2 in range(IN2_LIMIT_INF,IN2_LIMIT_SUP,stride2):
            out1 = (in1-k1+padding)//stride1
            out2 = (in2-k2+padding)//stride2
            dw[batch,filter,k1,k2,dim] += \
                ym1[batch,in1,in2,dim]* \
                sigmaOut[batch,out1,out2,filter]* \
                dphi[batch,out1,out2,filter]


#input1 = output1*stride1+k1-padding 
#output1 = (input1+padding-k1)/stride1
def computeLocalGradient(globalIndex,
                            sigmaOut,
                            outSize1,
                            outSize2,
                            filters,
                            inSize1,
                            inSize2,
                            inSize3,
                            kernel1,
                            kernel2,
                            stride1,
                            stride2,
                            padding,
                            sigmaIn,
                            dphi,
                            w):
    batch = globalIndex[0]
    in1 = globalIndex[1]//inSize2
    in2 = globalIndex[1]%inSize2
    in3 = globalIndex[2]

    K1_LIMIT_INF = max(0,in1-outSize1*stride1+padding+1)+in1%stride1
    K2_LIMIT_INF = max(0,in2-outSize2*stride2+padding+1)+in2%stride2
    K1_LIMIT_SUP = min(kernel1,in1+padding+1)
    K2_LIMIT_SUP = min(kernel2,in2+padding+1)

    sigmaIn[batch,in1,in2,in3] = 0
    for k1 in range(K1_LIMIT_INF,K1_LIMIT_SUP,stride1):
        for k2 in range(K2_LIMIT_INF,K2_LIMIT_SUP,stride2):
            for out3 in range(filters):
                out1 = (in1-k1+padding)//stride1
                out2 = (in2-k2+padding)//stride2
                sigmaIn[batch,in1,in2,in3] += w[out3,k1,k2,in3] \
                    *sigmaOut[batch,out1,out2,out3] \
                        *dphi[batch,out1,out2,out3]

     
def learningRule(globalIndex,
                lrate,
                filters,
                batchSize,
                wsize,
                dw,
                db,
                w,
                b):
    ind = globalIndex[0]

    sum = 0
    for batch in range(batchSize):
        index = (batch,*np.unravel_index(ind,w.shape))
        sum += dw[index]
    w[np.unravel_index(ind,w.shape)] += -sum*lrate/batchSize

    if(ind % wsize == 0):
        sum = 0
        j = ind//wsize
        for batch in range(batchSize):
            sum += db[batch,j]
        b[j,] += -sum*lrate/batchSize


def sigmoidTest(globalIndex,
            v,
            y, 
            dphi):
    ind = np.unravel_index(globalIndex[0],v.shape)
    if (v[ind] > 80):
        v[ind] = 80
    
    if (v[ind] < -80):
        v[ind] = -80
    
    y[ind] = 1/( 1 + np.exp(-v[ind]) )
    dphi[ind] = y[ind]*(1-y[ind])

def computeError(globalIndex,
                Y,
                e,
                y):
    ind = np.unravel_index(globalIndex[0],y.shape)
    e[ind] = -(Y[ind]-y[ind])
