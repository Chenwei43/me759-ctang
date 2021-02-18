#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include <cuda.h>
#include "cuda_runtime.h"
#include "vscale.cuh"

int main(int argc, char* argv[]) {
    unsigned int j;
	std::istringstream nn(argv[1]);
	if (nn >> j && nn.eof()) {

        unsigned int n = pow(2,j);		
        float* a_h = new float[n];
        float* b_h = new float[n];

        
        int my_seed = 137;
        std::mt19937 generator(my_seed);
        std::uniform_real_distribution<float> dist10(-10,10);
        std::uniform_real_distribution<float> dist1(0,1);
        for (unsigned int i = 0; i < n; i++) {
            a_h[i] = dist10(generator);
            b_h[i] = dist1(generator);
        }

        const int numThreads = 512;
        const int numBlocks = (n+numThreads-1)/numThreads;
        float *a, *b;
        cudaMalloc((void**)&a, sizeof(float) * n);
        cudaMalloc((void**)&b, sizeof(float) * n);
        cudaMemcpy(a, a_h, sizeof(float) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(b, b_h, sizeof(float) * n, cudaMemcpyHostToDevice);

        cudaEvent_t startEvent, stopEvent; 
        cudaEventCreate(&startEvent); 
        cudaEventCreate(&stopEvent);    
        cudaEventRecord(startEvent, 0);

        vscale<<<numBlocks,numThreads>>>(a, b, n);
        
        cudaEventRecord(stopEvent, 0); 
        cudaEventSynchronize(stopEvent); 
        float elapsedTime; 
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);      
        cudaEventDestroy(startEvent); 
        cudaEventDestroy(stopEvent);  

        cudaMemcpy(b_h, b, sizeof(float) * n, cudaMemcpyDeviceToHost);  

        std::cout << elapsedTime << "\n" ;
        std::cout << b_h[0] << "\n" ;
        std::cout << b_h[n-1] << "\n" ;

        cudaFree(a);
        cudaFree(b);
        delete []a_h;
        delete []b_h;   
    }
    return 0;
}
