kernel void forwardPropagate(global float *ym1,
                             const uint inSize,
                             const uint outSize,
                             global float *v,
                             global float *w,
                             global float *b){

    uint batch = get_global_id(0);
    uint j = get_global_id(1);

    v[outSize*batch + j] = b[j];
    for (unsigned int k = 0; k < inSize; k++){
        v[batch*outSize + j] += 
        ym1[batch*inSize + k] * 
        w[k*outSize + j];
    }
    
}

kernel void sigmoid(const uint outSize,
                    global float *v,
                    global float *y,
                    global float *dphi){
    
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
    //db[ind]= -e[ind]*dphi[ind];
}

kernel void computedb(global float *sigma,
                             global float *dphi,
                             global float *db){
    uint ind = get_global_id(0);
    db[ind]= sigma[ind]*dphi[ind];
}

kernel void computeGradients(global float *ym1,
                             const uint inSize,
                             const uint outSize,
                             global float *dw,
                             global float *db){
    uint batch = get_global_id(0);
    uint i = get_global_id(1);
    uint j = get_global_id(2);

    dw[batch*outSize*inSize + i*outSize + j] = 
            ym1[inSize*batch + i]*
            db[outSize*batch + j];
}

kernel void computeLocalGradient(const uint inSize,
                            const uint outSize,
                            global float *sigma,
                            global float *db,
                            global float *dphi,
                            global float *w){
    uint batch = get_global_id(0);
    uint i = get_global_id(1);
    uint indIn = inSize*batch + i;
    
    sigma[indIn] = 0;
    for(uint j = 0; j<outSize; j++){
        uint indOut = outSize*batch + j;
        sigma[indIn] += w[outSize*i+j]*db[indOut];
    }
}



kernel void learningRule(const float lrate,
                        const uint inSize,
                        const uint outSize,
                        const uint batchSize,
                        global float *dw,
                        global float *db,
                        global float *w,
                        global float *b){

    uint i = get_global_id(0);
    uint j = get_global_id(1);

    float Dw = 0;
    for(uint batch=0; batch<batchSize; batch++){
        Dw += dw[batch*outSize*inSize + i*outSize + j];
    }
    w[outSize*i+j] += -Dw*lrate/batchSize;

    if(i == 0){
        float Db = 0;
        for(uint batch=0; batch<batchSize; batch++){
            Db += db[outSize*batch + j];
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