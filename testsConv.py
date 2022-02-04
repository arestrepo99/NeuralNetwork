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
                    v,
                    w,
                    b):
    batch = globalIndex[0]
    filter = globalIndex[1]
    output1 = globalIndex[2]%outSize1
    output2 = globalIndex[2]//outSize1
    
    v[batch,output1,output2,filter] = b[filter]
    for k1 in range(kernel1):
        for k2 in range(kernel2):
            for dim in range(inSize3):
                v[batch,output1,output2,filter] += \
                ym1[batch,output1*stride1+k1,output2*stride2+k2,dim] * \
                w[filter, k1, k2,dim]


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

def computedb(globalIndex,
             sigma,
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
                sigma[np.unravel_index(indOut+indOut2,sigma.shape)] \
                *dphi[np.unravel_index(indOut+indOut2,dphi.shape)]

def computeGradients(globalIndex,
                        ym1,
                        sigma,
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
                        dw,
                        dphi
                        ):
    batch = globalIndex[0]//filters
    filter = globalIndex[0]%filters
    k1 = globalIndex[1]//kernel2
    k2 = globalIndex[1]%kernel2
    dim = globalIndex[2]
    dw[batch,filter,k1,k2,dim] = 0
    for output1 in range(outSize1):
        for output2 in range(outSize2):
                dw[batch,filter,k1,k2,dim] += \
                    ym1[batch,output1*stride1+k1,output2*stride2+k2,dim]* \
                    sigma[batch,output1,output2,filter]* \
                    dphi[batch,output1,output2,filter]

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
                            sigmaIn,
                            dphi,
                            w):
    batch = globalIndex[0]
    in1 = globalIndex[1]//inSize2
    in2 = globalIndex[1]%inSize2
    in3 = globalIndex[2]

    out1 = in1//stride1
    out2 = in2//stride2

    phase1 = in1%stride1
    phase2 = in2%stride2

    sigmaIn[batch,in1,in2,in3] = 0
    for k1 in range(max(phase1+inSize1,in1+kernel1+1)-inSize1,
                    kernel1+1+in1 - max(kernel1,1+in1),
                    stride1):
        for k2 in range(max(phase2+inSize2,in2+kernel2+1)-inSize2,
                        kernel1+1+in2 - max(kernel1,1+in2),
                        stride2):
            for out3 in range(filters):
                assert in1-k1>=0
                assert in2-k2>=0
                ind1 = (in1-k1)//stride1
                ind2 = (in2-k2)//stride2
                sigmaIn[batch,in1,in2,in3] += w[out3,k1,k2,in3] \
                    *sigmaOut[batch,ind1,ind2,out3] \
                        *dphi[batch,ind1,ind2,out3]

     
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
        b[j] += -sum*lrate/batchSize