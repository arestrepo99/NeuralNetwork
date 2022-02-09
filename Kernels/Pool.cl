kernel void averagepool(global float *ym1,
                             const int outSize1,
                             const int outSize2,
                             const int outSize3,
                             const int stride1,
                             const int stride2,
                             const int pool1,
                             const int pool2,
                             const int inSize1,
                             const int inSize2,
                             const int inSize3,
                             global float *y){

    int batch = get_global_id(0);
    int out1 = get_global_id(1)%outSize1;
    int out2 = get_global_id(1)/outSize1;
    int out3 = get_global_id(2);

    int indOut = batch       *outSize3*outSize2*outSize1 + 
                out1        *outSize3*outSize2 + 
                out2        *outSize3 + 
                out3      ;
    int indIn = batch              *inSize3*inSize2*inSize1 + 
                out1*stride1       *inSize3*inSize2 +
                out2*stride2       *inSize3;

    y[indOut] = 0;
    for(int p1 = 0; p1<pool1; p1++){
        int indIn1 = p1 *inSize3*inSize2;
        for(int p2 = 0; p2<pool2; p2++){
            int indIn2 = p2 *inSize3;
            y[indOut] += 
            ym1[indIn + indIn1 + indIn2 + out3];
        }
    }
    y[indOut] *= 1/(pool1*pool2);
}

kernel void averagepoolgrad(global float *sigmaOut,
                            const int outSize1,
                            const int outSize2,
                            const int outSize3,
                            const int inSize1,
                            const int inSize2,
                            const int inSize3,
                            const int pool1,
                            const int pool2,
                            const int stride1,
                            const int stride2,
                            global float *sigmaIn){
                                
    int batch = get_global_id(0);
    int in1 = get_global_id(1)/inSize2;
    int in2 = get_global_id(1)%inSize2;
    int in3 = get_global_id(2);
  
    int indIn = batch               *inSize3*inSize2*inSize1 + 
                 in1                  *inSize3*inSize2 + 
                 in2                  *inSize3+
                 in3;  
    
    int indOut = batch         *outSize3*outSize2*outSize1;
    int indOut1;
    int indOut2;
    
    sigmaIn[indIn] = 0;
    for(int p1 = 0; p1<pool1; p1+=stride1){
        indOut1 = (in1+p1)/stride1    *outSize3*outSize2;
        for(int p2 =0; p2<pool2; p2+=stride2){
            indOut2 = (in2+p2)/stride2    *outSize3;
            sigmaIn[indIn] += sigmaOut[indOut+indOut1+indOut2+in3];         
        }
    }
    sigmaIn[indIn] *= 1/pool1*pool2;
}