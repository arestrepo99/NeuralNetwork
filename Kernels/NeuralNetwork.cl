kernel void computeError(global float *Y,
                             global float *e,
                             global float *y){
    uint ind = get_global_id(0);
    e[ind] = -(Y[ind]-y[ind]);
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
    E[ind] = E[ind];
}
