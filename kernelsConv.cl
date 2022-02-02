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

kernel void sigmoid(global float *v,
                    global float *y, global float *dphi){
 
    uint ind = get_global_id(0);
    if (v[ind] > 80){
        v[ind] = 80;
    }
    if (v[ind] < -80){
        v[ind] = -80;
    }
    y[ind] = 1/( 1 + exp(-v[ind]) );
    dphi[ind] = y[ind]*(1-y[ind]);
}


kernel void computeError(global float *Y,
                             global float *e,
                             global float *y){

    uint ind = get_global_id(0);
    e[ind] = -(Y[ind]-y[ind]); //multiplied by -1 to substitute sigma
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
            ind2 = o1*outSize1*filters +  o2*filters;
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
    uint k1 = get_global_id(1)%kernel2;
    uint k2 = get_global_id(1)/kernel2;
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
                                
    uint batch = get_global_id(0)/filters;
    uint in1 = get_global_id(1)/inSize2;
    uint in2 = get_global_id(1)%inSize2;
    uint dim = get_global_id(2);
    
    uint indIn = batch               *inSize3*inSize2*inSize1 + 
                 in1                  *inSize3*inSize2 + 
                 in2                  *inSize3+
                 dim;  
    
    uint out1 = in1/stride1;
    uint out2 = in2/stride2;

    uint indOut = batch         *filters*outSize2*outSize1 +
            out1                *filters*outSize2 + 
            out2                *filters;

    uint indOut2;
    sigmaIn[indIn] = 0;
    for(uint k1 = 0; k1<kernel1; k1+=stride1){
        for(uint k2 = 0; k2<kernel2; k2+=stride2){
            for(uint filter = 0; filter<filters; filter++){
                if(out1>=k1 && out1<=outSize1+k1 && out2>=k2 && out2<=outSize2+k2){
                    indOut2 =
                    -k1    *filters*outSize2 + 
                    -k2    *filters + 
                    filter;
                    sigmaIn[indIn] +=
                    w[ filter      *kernel1*kernel2*inSize3 +
                            k1          *kernel2*inSize3 +
                            k2          *inSize3 +
                            dim]
                        *sigmaOut[indOut+indOut2]*dphi[indOut+indOut2];
                }
            }
        }
    }
}
 