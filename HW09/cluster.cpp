#include <cstdlib>
#include <iostream>
#include "cluster.h"


//solution 1- reduction
void cluster(const size_t n, const size_t t, const int *arr, const int *centers, int *dists) {
    
    #pragma omp parallel num_threads(t)
    {   
        //int dist_temp=0;
        unsigned int tid = omp_get_thread_num();
        
        #pragma omp for reduction(+:dists[tid])
            for (size_t i = 0; i < n; i++) {
                dists[tid] += abs(arr[i] - centers[tid]);
            }
        
    }
}

/*
// solution 2
void cluster(const size_t n, const size_t t, const int *arr, const int *centers, int *dists) {
    
    #pragma omp parallel num_threads(t)
    {   
        int dist_temp=0;
        unsigned int tid = omp_get_thread_num();
        
        #pragma omp for private(dist_temp)
            for (size_t i = 0; i < n; i++) {
                dist_temp += abs(arr[i] - centers[tid]);
            }

            dists[tid] = dist_temp;
    }
}
*/
/*
// original- with false sharing
void cluster(const size_t n, const size_t t, const int *arr, const int *centers, int *dists) {
    #pragma omp parallel num_threads(t)
    {
        unsigned int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n; i++) {
            dists[tid] += abs(arr[i] - centers[tid]);
        }
    }
}
*/