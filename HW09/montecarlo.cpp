#include <cstdlib>
#include <iostream>
#include <cmath>
#include "montecarlo.h"


// this function returns the number of points that lay inside
// a circle using OpenMP parallel for. 
// You also need to use the simd directive.

// x - an array of random floats in the range [-radius, radius] with length n.
// y - another array of random floats in the range [-radius, radius] with length n.
int montecarlo(const size_t n, const float *x, const float *y, const float radius){
    
    int count=0;
    #pragma omp parallel for simd reduction(+:count)
        for (size_t i=0; i<n; i++){
            float dist = pow(x[i],2)+pow(y[i],2)-pow(radius,2);
            if (dist<0){
                count += 1;
            }else{
                count += 0;
            }
        }

    return count;
}