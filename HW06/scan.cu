#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "scan.cuh"


__global__ void hillis_steele_blocks(float* g_odata, const float* g_idata, float *eob) {
    /* **inclusive** scan of each block */

    extern volatile __shared__  float temp[]; // allocated on invocation

    int thid = threadIdx.x;
    int idxFull = threadIdx.x + blockIdx.x * blockDim.x;


        int pout = 0, pin = 1;

        // load input into shared memory. when overshoot, write 0s. 
        temp[thid] = g_idata[idxFull];
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            pout = 1 - pout; // swap double buffer indices
            pin = 1 - pout;

            if (thid >= offset)
                temp[pout * blockDim.x + thid] = temp[pin * blockDim.x + thid] + temp[pin * blockDim.x + thid - offset];
            else
                temp[pout * blockDim.x + thid] = temp[pin * blockDim.x + thid];

            __syncthreads(); // I need this here before I start next iteration 
        }

        // crop the padded
            g_odata[idxFull] = temp[pout * blockDim.x + thid];
    
        __syncthreads();

        eob[blockIdx.x] = g_odata[blockDim.x * (1+blockIdx.x)-1];
    
    
}

__global__ void hillis_steele_eob(float* g_odata, const float* g_idata, unsigned int n) {
    /* **inclusive** scan of array consists of last entries of each block 
        Under the assumption that length of input <= threads_per_block**2, this can be done safely by 1 block.
    */

    extern volatile __shared__  float temp[]; // allocated on invocation

    int thid = threadIdx.x;
    int pout = 0, pin = 1;


    // load input into shared memory. 
    temp[thid] = g_idata[thid];
 
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;

        if (thid >= offset)
            temp[pout * blockDim.x + thid] = temp[pin * blockDim.x + thid] + temp[pin * blockDim.x + thid - offset];
        else
            temp[pout * blockDim.x + thid] = temp[pin * blockDim.x + thid];

        __syncthreads(); // I need this here before I start next iteration 
    }
    
    g_odata[thid] = temp[pout * blockDim.x + thid];
}

/* add scanned eob to individually scanned blocks -> final results */
__global__ void add_eob(float* g_odata, float* eob, unsigned int threads_per_block, unsigned int numBlocks) {

    if (blockIdx.x < (numBlocks - 1)) {
        g_odata[(blockIdx.x+1)*threads_per_block+threadIdx.x] += eob[blockIdx.x];
    }    

}

/* Added last entry of scanned eob to entries of residual block*/
__global__ void add_lastFull(float* g_odata, float* eob, unsigned int num_fullBlocks) {

    g_odata[threadIdx.x] += eob[num_fullBlocks-1];    

}


__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {

    
    unsigned int num_entries_last_block = n % threads_per_block;
    if (num_entries_last_block == 0) {

        //scan each block
        unsigned int numBlocks = (n + threads_per_block - 1) / threads_per_block;
        
        float* end_of_block, * end_of_block_scanned;
        cudaMalloc((void**)&end_of_block, numBlocks * sizeof(float));
        cudaMalloc((void**)&end_of_block_scanned, numBlocks * sizeof(float));

        hillis_steele_blocks<<<numBlocks, threads_per_block, 2 * threads_per_block * sizeof(float)>>> (output, input, end_of_block);

        //scan array composed of the last entries of each block
        //under the assumption, it can be done with a single block
        // unsigned int numBlocks_scanBlock = (numBlocks + threads_per_block - 1) / (2 * threads_per_block);
        hillis_steele_eob<<<1, numBlocks, 2 * numBlocks * sizeof(float)>>>(end_of_block_scanned, end_of_block, numBlocks);


        //Add end of each block to corresponding entries
        add_eob<<<numBlocks, threads_per_block >>>(output, end_of_block_scanned, threads_per_block, numBlocks);


        cudaFree(end_of_block);
        cudaFree(end_of_block_scanned);
    }
    else
    {
        unsigned int real_n = n - num_entries_last_block;
        unsigned int num_fullBlocks = (real_n + threads_per_block - 1) / threads_per_block;

        // scan the first real_n full blocks
        float* end_of_block, * end_of_block_scanned;
        cudaMalloc((void**)&end_of_block, num_fullBlocks * sizeof(float));
        cudaMalloc((void**)&end_of_block_scanned, num_fullBlocks * sizeof(float));
        hillis_steele_blocks<<<num_fullBlocks, threads_per_block, 2 * threads_per_block * sizeof(float)>>>(output, input, end_of_block);
        hillis_steele_eob<<<1, num_fullBlocks, 2 * num_fullBlocks * sizeof(float)>>>(end_of_block_scanned, end_of_block, num_fullBlocks);
        add_eob<<<num_fullBlocks, threads_per_block>>>(output, end_of_block_scanned, threads_per_block, num_fullBlocks);

        // scan the last block
        float* last_block_scanned = &(output[real_n]);
        const float* last_block = &(input[real_n]);        
        hillis_steele_eob<<<1, num_entries_last_block, 2 * num_entries_last_block * sizeof(float)>>>(last_block_scanned, last_block, num_entries_last_block);

        // Add last entry of  end_of_block_scanned to last_block_scanned -> the residual block is scanned. Then stitch full blocks and residual block.
        add_lastFull<<<1, num_entries_last_block>>>(last_block_scanned, end_of_block_scanned, num_fullBlocks);

        cudaFree(end_of_block);
        cudaFree(end_of_block_scanned);

    }  
    cudaDeviceSynchronize();

}