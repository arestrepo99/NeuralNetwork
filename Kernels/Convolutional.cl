kernel void forwardPropagate(__global float *ym1,
                             __global float *v,
                             __global float *w,
                             __global float *b,
                             const int outSize1,
                             const int outSize2,
                             const int filters,
                             const int stride1,
                             const int stride2,
                             const int kernel1,
                             const int kernel2,
                             const int inSize1,
                             const int inSize2,
                             const int inSize3,
                             const int padding){

    int batch = get_global_id(0);
    int filter = get_global_id(1);
    int out1 = get_global_id(2)%outSize1;
    int out2 = get_global_id(2)/outSize1;
    
    int indw = filter*kernel1*kernel2*inSize3;
    
    int indOut = batch       *filters*outSize2*outSize1 + 
                out1     *filters*outSize2 + 
                out2     *filters + 
                filter      ;

    const int in1 = out1*stride1-padding;
    const int in2 = out2*stride2-padding;

    int indIn = batch               *inSize3*inSize2*inSize1 + 
                in1                 *inSize3*inSize2 +
                in2                 *inSize3;
    
    int indIn1; int indIn2; int indw1; int indw2;
    const int K1_LIMIT = min(kernel1,inSize1-in1);
    const int K2_LIMIT = min(kernel2,inSize2-in2);
    v[indOut] = b[filter];
    for(int k1 = max(0,-in1); k1<K1_LIMIT; k1++){
        indIn1 = k1 *inSize3*inSize2;
        indw1 = k1 *kernel2*inSize3;
        for(int k2 = max(0,-in2); k2<K2_LIMIT; k2++){
            for(int dim = 0; dim<inSize3; dim++){
                indIn2 =  k2 *inSize3;
                indw2 = k2 *inSize3;

                v[indOut] += 
                ym1[indIn + indIn1 + indIn2 + dim] * w[indw + indw1+ indw2 + dim];
            }
        }
    }
}


kernel void computedb(global float *sigmaOut,
                             global float *dphi,
                             global float *db,
                             const uint outSize1,
                             const uint outSize2,
                             const uint filters){
    uint batch = get_global_id(0);
    uint filter = get_global_id(1);

    uint ind = batch*filters+filter;
    uint indOut = batch*filters*outSize1*outSize2+filter;
    db[ind] = 0;
    uint indOut1;
    uint indOut2;
    for(uint out1 = 0; out1<outSize1; out1++){
        indOut1 = out1*outSize2*filters;
        for(uint out2 = 0; out2<outSize2; out2++){
            indOut2 = out2*filters;
            db[ind] += sigmaOut[indOut+indOut1+indOut2]*dphi[indOut+indOut1+indOut2];
        }
    }
}



kernel void computeGradients(global float * ym1,
                        global float *sigmaOut,
                        global float *dw,
                        global float *dphi,
                        const int outSize1,
                        const int outSize2,
                        const int filters,
                        const int inSize1,
                        const int inSize2,
                        const int inSize3,
                        const int kernel1,
                        const int kernel2,
                        const int stride1,
                        const int stride2,
                        const int padding
                        ){
    int batch = get_global_id(0)/filters;
    int filter = get_global_id(0)%filters;
    int k1 = get_global_id(1)/kernel2;
    int k2 = get_global_id(1)%kernel2;
    int dim = get_global_id(2);
    
    const int IN1_LIMIT_INF = max(0,k1-padding) +(stride1-max(0,-k1+padding))%stride1;
    const int IN2_LIMIT_INF = max(0,k2-padding) +(stride2-max(0,-k2+padding))%stride2;
    const int IN1_LIMIT_SUP = min(inSize1,outSize1*stride1-padding+k1);
    const int IN2_LIMIT_SUP = min(inSize2,outSize2*stride2-padding+k2);

    int wind = batch         *filters*kernel1*kernel2*inSize3 +
                 filter      *kernel1*kernel2*inSize3 +
                 k1          *kernel2*inSize3 +
                 k2          *inSize3 + 
                 dim;

    int indIn = batch        *inSize3*inSize2*inSize1 + 
                 dim;  

    int indOut = batch       *filters*outSize2*outSize1 + 
                 filter;

    int indIn1;
    int indIn2;
    int indOut1;
    int indOut2;
    dw[wind] = 0;
    for(int in1 = IN1_LIMIT_INF; in1<IN1_LIMIT_SUP; in1+=stride1){
        indIn1 =  in1 * inSize3*inSize2;
        indOut1 = (in1-k1+padding)/stride1 * filters*outSize2;
        for(int in2 = IN2_LIMIT_INF; in2<IN2_LIMIT_SUP; in2+=stride2){
            indIn2 = in2 *inSize3;
            indOut2 = (in2-k2+padding)/stride2 * filters;
            dw[wind] += ym1[indIn+indIn1+indIn2]*
                sigmaOut[indOut+indOut1+indOut2]*
                dphi[indOut+indOut1+indOut2];
        }
    } 
}

kernel void computeLocalGradient(global float *sigmaOut,
                            global float *sigmaIn,
                            global float *dphi,
                            global float *w,
                            const int outSize1,
                            const int outSize2,
                            const int filters,
                            const int inSize1,
                            const int inSize2,
                            const int inSize3,
                            const int kernel1,
                            const int kernel2,
                            const int stride1,
                            const int stride2,
                            const int padding
                            ){
                                
    int batch = get_global_id(0);
    int in1 = get_global_id(1)/inSize2;
    int in2 = get_global_id(1)%inSize2;
    int in3 = get_global_id(2);
    
    const int K1_LIMIT_INF = max(0,in1-outSize1*stride1+padding+1)+in1%stride1;
    const int K2_LIMIT_INF = max(0,in2-outSize2*stride2+padding+1)+in2%stride2;
    const int K1_LIMIT_SUP = min(kernel1,in1+padding+1);
    const int K2_LIMIT_SUP = min(kernel2,in2+padding+1);

    int indIn = batch               *inSize3*inSize2*inSize1 + 
                 in1                  *inSize3*inSize2 + 
                 in2                  *inSize3+
                 in3;  
    
    int indOut = batch         *filters*outSize2*outSize1;
    int indOut1;
    int indOut2;
    
    
    sigmaIn[indIn] = 0;
    for(int k1 = K1_LIMIT_INF; k1<K1_LIMIT_SUP; k1+=stride1){
        indOut1 = (in1+padding-k1)/stride1    *filters*outSize2;
        for(int k2 =K2_LIMIT_INF; k2<K2_LIMIT_SUP; k2+=stride2){
            indOut2 = (in2+padding-k2)/stride2    *filters;
            for(int indOut3 = 0; indOut3<filters; indOut3++){
                sigmaIn[indIn] += 
                    w[ indOut3      *kernel1*kernel2*inSize3 +
                        k1          *kernel2*inSize3 +
                        k2          *inSize3 +
                        in3] *
                    sigmaOut[indOut+indOut1+indOut2+indOut3]*
                    dphi[indOut+indOut1+indOut2+indOut3];         
            }
        }
    }
}
 
 
kernel void learningRule(const float lrate,
                        global float *dw,
                        global float *db,
                        global float *w,
                        global float *b,     
                        const uint filters,
                        const uint batchSize,
                        const uint wsize ){

    uint ind = get_global_id(0);

    float sum = 0;
    for(uint batch=0; batch<batchSize; batch++){
        sum += dw[batch*filters*wsize + ind];
    }
    w[ind] += -sum*lrate /batchSize;

    if(ind % wsize == 0){
        sum = 0;
        uint j = ind/wsize;
        for(uint batch=0; batch<batchSize; batch++){
            sum += db[batch*filters + j];
        }
        b[j] += -sum*lrate/batchSize;
    }
}