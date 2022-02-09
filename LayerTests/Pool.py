def averagepool(globalIndex,
                ym1,
                outSize1,
                outSize2,
                outSize3,
                stride1,
                stride2,
                pool1,
                pool2,
                inSize1,
                inSize2,
                inSize3,
                y):

    batch = globalIndex[0]
    out1 = globalIndex[1]%outSize1
    out2 = globalIndex[1]//outSize1
    out3 = globalIndex[2]

    y[batch, out1, out2, out3] = 0
    for p1 in range(pool1):
        for p2 in range(pool2):
            y[batch,out1,out2,out3] += \
            ym1[batch, out1*stride1 +p1, out2*stride2+p2,out3]
    y[batch,out1,out2,out3] *= 1/(pool1*pool2)


def averagepoolgrad(globalIndex,
                        sigmaOut,
                        outSize1,
                        outSize2,
                        outSize3,
                        inSize1,
                        inSize2,
                        inSize3,
                        pool1,
                        pool2,
                        stride1,
                        stride2,
                        sigmaIn):
                                
    batch = globalIndex[0]
    in1 = globalIndex[1]//inSize2
    in2 = globalIndex[1]%inSize2
    in3 = globalIndex[2]

    sigmaIn[batch,in1,in2,in3] = 0
    for p1 in range(pool1):
        for p2 in range(pool2):
                sigmaIn[batch,in1,in2,in3] += sigmaOut[batch,(in1+p1)/stride1,(in2+p2)/stride2,in3]    
                     
    sigmaIn[batch,in1,in2,in3] *= 1/pool1*pool2


