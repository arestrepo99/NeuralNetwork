# Testing kernels

def testComputeLocalGradient(conv, sigmaOut):
    sigmaIn = conv.sigma.get()
    dphi = conv.dphi.get()
    w = conv.w.get()

    batch_size = conv.batchSize
    outSize1,outSize2,filters = conv.outputShape
    inSize1,inSize2,inSize3 = conv.inputShape
    kernel1,kernel2 = conv.kernel
    stride1,stride2 = conv.strides
    def computeLocalGradient(batch,in1,in2,in3):
        indIn = (batch,in1,in2,in3)
        out1 = in1*stride1
        out2 = in2*stride2

        sigmaIn[indIn] = 0
        range1 = range(max(inSize1,kernel1+in1)-inSize1,min(kernel1,out1+1),stride1)
        range2 = range(max(inSize2,kernel2+in2)-inSize2,min(kernel2,out2+1),stride2)
        for k1 in range1:
            for k2 in range2:
                for out3 in range(filters):
                    assert out1-k1>=0
                    assert out2-k2>=0
                    sigmaOut[batch,out1-k1,out2-k2,out3] = out1-k1
                    sigmaIn[indIn] +=1
                        
    for batch in range(batch_size):
        for in1 in range(inSize1):
            for in2 in range(inSize2):
                for in3 in range(inSize3):
                    computeLocalGradient(batch,in1,in2,in3)