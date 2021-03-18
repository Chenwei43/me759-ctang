#include <cuda.h>
#include <iostream>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "count.cuh"


int get_num_uniques(thrust::device_vector<int>& input){
    return thrust::inner_product(input.begin(), input.end() - 1, 
                                input.begin() + 1, 0, thrust::plus<int>(), thrust::not_equal_to<int>()) + 1;

}

void count(const thrust::device_vector<int>& d_in, thrust::device_vector<int>& values, thrust::device_vector<int>& counts){

    thrust::device_vector<int> d_in_sorted = d_in;
    thrust::sort(d_in_sorted.begin(), d_in_sorted.end());
    size_t numUniques = get_num_uniques(d_in_sorted);
    values.resize(numUniques);
    counts.resize(numUniques);
    thrust::device_vector<int> temp(d_in.size(), 1);

    thrust::reduce_by_key(d_in_sorted.begin(), d_in_sorted.end(), temp.begin(), values.begin(), counts.begin());

}


