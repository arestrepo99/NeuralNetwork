kernel void forwardPropagate(global float *ym1,
                             global float *v,
                             __constant float *w,
                             global float *b,
                             const uint inSize,
                             const uint outSize){

    uint batch = get_global_id(0);
    uint j = get_global_id(1);

    uint indOut = batch*outSize + j;
    uint indIn = batch*inSize;
    float acc = b[j];
    for (unsigned int k = 0; k < inSize; k++){
        acc += 
        ym1[indIn + k] * 
        w[k*outSize + j];
    }
    v[outSize*batch + j] = acc;
}


kernel void computedb(global float *sigmaOut,
                             global float *dphi,
                             global float *db){
    uint ind = get_global_id(0);
    db[ind]= sigmaOut[ind]*dphi[ind];
}

kernel void computeGradients(global float *ym1,
                             global float *dw,
                             global float *db,
                             const uint inSize,
                             const uint outSize){
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
                            const uint outSize){

    uint batch = get_global_id(0);
    uint i = get_global_id(1);
    uint indIn = inSize*batch + i;
    uint indOut = outSize*batch;
    sigma[indIn] = 0;
    for(uint j = 0; j<outSize; j++){;
        sigma[indIn] += w[outSize*i+j]*db[indOut+j];
    }
}



kernel void learningRule(const float lrate,
                        global float *dw,
                        global float *db,
                        global float *w,
                        global float *b,
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
    w[outSize*i+j] += -Dw*lrate;

    if(i == 0){
        float Db = 0;
        ind = j;
        for(uint batch=0; batch<batchSize; batch++){
            ind += outSize;
            Db += db[ind];
        }
        b[j] += -Db*lrate;
    }
}
