kernel void sigmoid(global float *v,
                    global float *y,
                    global float *dphi){
    
    uint ind = get_global_id(0);
    
    if (v[ind] > 20){
        v[ind] = 20;
    }
    if (v[ind] < -20){
        v[ind] = -20;
    }

    y[ind] = 1/( 1 + exp(-v[ind]));
    dphi[ind] = y[ind]*(1-y[ind]);
}