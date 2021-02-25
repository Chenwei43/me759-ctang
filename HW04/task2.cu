#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stencil.cuh"

int main(int argc, char* argv[]) {
    unsigned int n;
    unsigned int threads_per_block;
    unsigned int R;
    std::istringstream input_1(argv[1]);
    std::istringstream input_2(argv[2]);
    std::istringstream input_3(argv[3]);
    if (input_1 >> n && input_2>>R && input_3>>threads_per_block && input_3.eof()) {
        //create arrays on unified mem
		float* image, * mask;
		cudaMallocManaged(&image, sizeof(float) * n);
		cudaMallocManaged(&mask, sizeof(float) * (2 * R + 1));

		std::random_device entropy_source;
		std::mt19937 generator(entropy_source());
		std::uniform_real_distribution<float> dist(-1.0, 1.0);
		for (unsigned int i = 0; i < n; i++) {
			image[i] = dist(generator);			
		}
		for (unsigned int i = 0; i < (2 * R + 1); i++) {
			mask[i] = dist(generator);
		}

		float* output;
		cudaMallocManaged(&output, sizeof(float) * n);

		// kernel call
		cudaEvent_t startEvent, stopEvent;
		cudaEventCreate(&startEvent);
		cudaEventCreate(&stopEvent);
		cudaEventRecord(startEvent, 0);

		stencil(image, mask, output, n, R, threads_per_block);

		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		cudaEventDestroy(startEvent);
		cudaEventDestroy(stopEvent);

		std::cout << output[n - 1] << "\n";
		std::cout << elapsedTime << "\n";	
		
		cudaFree(image);
		cudaFree(output);
		cudaFree(mask);
	}
	return 0;
}