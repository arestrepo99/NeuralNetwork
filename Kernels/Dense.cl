kernel void forwardPropagate(global float *ym1,
                             global float *v,
                             global float *w,
                             global float *b,
                             const uint inSize,
                             const uint outSize,){

    uint batch = get_global_id(0);
    uint j = get_global_id(1);

    uint indOut = batch*outSize + j;
    uint indIn = batch*inSize;
    v[outSize*batch + j] = b[j];
    for (unsigned int k = 0; k < inSize; k++){
        v[indOut] += 
        ym1[indIn + k] * 
        w[k*outSize + j];
    }
    
}

kernel void computeError(global float *Y,
                             global float *e,
                             global float *y){
    uint ind = get_global_id(0);
    e[ind] = -(Y[ind]-y[ind])/2;
}

kernel void computedb(global float *sigmaOut,
                             global float *dphi,
                             global float *db){
    uint ind = get_global_id(0);
    db[ind]= sigmaOut[ind]*dphi[ind];
}

kernel void computeGradients(global float *ym1,
                             global float *dw,
                             global float *db),
                             const uint inSize,
                             const uint outSize{
    uint batch = get_global_id(0);
    uint i = get_global_id(1);
    uint j = get_global_id(2);

    dw[batch*outSize*inSize + i*outSize + j] = 
            ym1[inSize*batch + i]*
            db[outSize*batch + j];
}

kernel void computeLocalGradient(global float *sigma,
                            global float *db,
                            global float *dphi,
                            global float *w,
                            const uint inSize,
                            const uint outSize,){
    uint batch = get_global_id(0);
    uint i = get_global_id(1);
    uint indIn = inSize*batch + i;
    uint indOut = outSize*batch;
    sigma[indIn] = 0;
    for(uint j = 0; j<outSize; j++){;
        sigma[indIn] += w[outSize*i+j]*db[indOut];
    }
}



kernel void learningRule(global float *dw,
                        global float *db,
                        global float *w,
                        global float *b,
                        const float lrate,
                        const uint inSize,
                        const uint outSize,
                        const uint batchSize){

    uint i = get_global_id(0);
    uint j = get_global_id(1);

    uint ind = i*outSize + j;
    float Dw = 0;
    for(uint batch=0; batch<batchSize; batch++){
        ind += outSize*inSize;
        Dw += dw[ind];
    }
    w[outSize*i+j] += -Dw*lrate/batchSize;

    if(i == 0){
        float Db = 0;
        ind = j;
        for(uint batch=0; batch<batchSize; batch++){
            ind += outSize;
            Db += db[ind];
        }
        b[j] += -Db*lrate/batchSize;
    }
}


kernel void squareError(global float *e,
                        global float *e2){

    uint ind = get_global_id(0);
    e2[ind] = pow(e[ind],2);
    }

kernel void meanError(const int batchSize,
                const int outSize,
                global float *e2,
                global float *E){
    uint ind = get_global_id(0);
    E[ind] = 0;            
    for(int batch=0; batch<batchSize; batch++){
        E[ind] += e2[batch*outSize + ind];
    }
    E[ind] = E[ind]/batchSize;
}