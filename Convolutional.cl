kernel void forwardPropagate(global float *ym1,
                             const uint outSize1,
                             const uint outSize2,
                             const uint filters,
                             const uint stride1,
                             const uint stride2,
                             const uint kernel1,
                             const uint kernel2,
                             const uint inSize1,
                             const uint inSize2,
                             const uint inSize3,
                             global float *v,
                             global float *w,
                             global float *b){

    uint batch = get_global_id(0);
    uint filter = get_global_id(1);
    uint output1 = get_global_id(2)%outSize1;
    uint output2 = get_global_id(2)/outSize1;
    
    uint wind = filter*kernel1*kernel2*inSize3;
    
    uint indOut = batch       *filters*outSize2*outSize1 + 
                output1     *filters*outSize2 + 
                output2     *filters + 
                filter      ;
    uint indIn = batch               *inSize3*inSize2*inSize1 + 
                output1*stride1     *inSize3*inSize2 +
                output2*stride2     *inSize3;
    
    v[indOut] = b[filter];
    for(uint k1 = 0; k1<kernel1; k1++){
        for(uint k2 = 0; k2<kernel2; k2++){
            for(uint dim = 0; dim<inSize3; dim++){
                v[indOut] += 
                ym1[indIn + k1*inSize3*inSize2 + k2 *inSize3 + dim] *
                w[wind + k1*kernel2*inSize3 + k2 * inSize3 + dim];

            }
        }
    }
}


kernel void computedb(global float *sigma,
                             const uint outSize1,
                             const uint outSize2,
                             const uint filters,
                             global float *dphi,
                             global float *db){
    uint batch = get_global_id(0);
    uint filter = get_global_id(1);

    uint ind = batch*filters+filter;
    uint indOut = batch*filters*outSize1*outSize2+filter;
    db[ind] = 0;
    uint ind2;
    for(uint o1 = 0; o1<outSize1; o1++){
        for(uint o2 = 0; o2<outSize2; o2++){
            ind2 = o1*outSize2*filters +  o2*filters;
            db[ind] += sigma[indOut+ind2]*dphi[indOut+ind2];
        }
    }
}

kernel void computeGradients(global float * ym1,
                        global float *sigma,
                        const uint outSize1,
                        const uint outSize2,
                        const uint filters,
                        const uint inSize1,
                        const uint inSize2,
                        const uint inSize3,
                        const uint kernel1,
                        const uint kernel2,
                        const uint stride1,
                        const uint stride2,
                        global float *dw,
                        global float *dphi
                        ){
    uint batch = get_global_id(0)/filters;
    uint filter = get_global_id(0)%filters;
    uint k1 = get_global_id(1)/kernel2;
    uint k2 = get_global_id(1)%kernel2;
    uint dim = get_global_id(2);
    
    uint wind = batch        *filters*kernel1*kernel2*inSize3 +
                 filter       *kernel1*kernel2*inSize3 +
                 k1          *kernel2*inSize3 +
                 k2          *inSize3 + 
                 dim;

    uint indIn = batch               *inSize3*inSize2*inSize1 + 
                 k1                  *inSize3*inSize2 + 
                 k2                  *inSize3+
                 dim;  

    uint indOut = batch       *filters*outSize2*outSize1 + 
                 filter      ;
    uint indIn2;
    uint indOut2;

    dw[wind] = 0;
    for(uint output1 = 0; output1<outSize1; output1++){
        for(uint output2 = 0; output2<outSize2; output2++){
                indIn2 = output1*stride1     *inSize3*inSize2 +
                    output2*stride2     *inSize3 ;
                indOut2 = output1     *filters*outSize2 + 
                    output2     *filters;
                dw[wind] += ym1[indIn+indIn2]*sigma[indOut+indOut2]*dphi[indOut+indOut2];
        }
    } 
}

kernel void computeLocalGradient(global float *sigmaOut,
                            const uint outSize1,
                            const uint outSize2,
                            const uint filters,
                            const uint inSize1,
                            const uint inSize2,
                            const uint inSize3,
                            const uint kernel1,
                            const uint kernel2,
                            const uint stride1,
                            const uint stride2,
                            global float *sigmaIn,
                            global float *dphi,
                            global float *w){
                                
    uint batch = get_global_id(0);
    uint in1 = get_global_id(1)/inSize2;
    uint in2 = get_global_id(1)%inSize2;
    uint in3 = get_global_id(2);
  
    uint indIn = batch               *inSize3*inSize2*inSize1 + 
                 in1                  *inSize3*inSize2 + 
                 in2                  *inSize3+
                 in3;  
    
    uint indOut = batch         *filters*outSize2*outSize1;
    uint indOut2;
    
    sigmaIn[indIn] = 0;
    for(uint k1 =in1%stride1; k1<kernel1; k1+=stride1){
        if ( in1>=k1 && in1+kernel1<k1+inSize1+1){ 
            for(uint k2 =in2%stride2; k2<kernel2; k2+=stride2){
                if(in2>=k2 && in2+kernel2<k2+inSize2+1) {
                    for(uint out3 = 0; out3<filters; out3++){
                        indOut2 =
                        (in1-k1)/stride1    *filters*outSize2 + 
                        (in2-k2)/stride2    *filters + 
                        out3;
                        sigmaIn[indIn] += 
                            w[ out3             *kernel1*kernel2*inSize3 +
                                k1     *kernel2*inSize3 +
                                k2     *inSize3 +
                                in3] *
                        sigmaOut[indOut+indOut2]*dphi[indOut+indOut2]; 
                    }
                }                
            }
        }
    }
}
 
 
kernel void learningRule(const float lrate,
                        const uint filters,
                        const uint batchSize,
                        const uint wsize,
                        global float *dw,
                        global float *db,
                        global float *w,
                        global float *b){

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