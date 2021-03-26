#include <stddef.h>
#include <stdlib.h>
#include <omp.h>
#include "matmul.h"

void mmul(const float* A, const float* B, float* C, const std::size_t n){

    #pragma omp for schedule(static)
    for (size_t i=0; i<n; i++){
        for (size_t k=0; k<n; k++){      
            for (size_t j=0; j<n; j++){
                C[i*n+j] += A[i*n+k]*B[j+k*n];
            }
        }
    }
}

