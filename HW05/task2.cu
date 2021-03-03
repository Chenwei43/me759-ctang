#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matmul.cuh"

int main(int argc, char* argv[])
{
    unsigned int n;
    unsigned int block_dim;
    std::istringstream input_1(argv[1]);
    std::istringstream input_2(argv[2]);
    if (input_1 >> n && input_2>>block_dim && input_2.eof()) {
                //create arrays on managed mem
        int* A, * B, * C;
        cudaMallocManaged(&A, sizeof(int) * n * n);
        cudaMallocManaged(&B, sizeof(int) * n * n);
        cudaMallocManaged(&C, sizeof(int) * n * n);

        std::random_device entropy_source;
        std::mt19937 generator(entropy_source());
        std::uniform_int_distribution<int> dist1(-10,10);
        for (unsigned int i = 0; i < n*n; i++) {
            A[i] = dist1(generator);
            B[i] = dist1(generator);

        }

        // kernel call
        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);

        matmul_1(A, B, C, n, block_dim);

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);

        std::cout << C[0] << "\n";
        std::cout << C[n * n - 1] << "\n";
        std::cout << elapsedTime << "\n";

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        
        //Float
        float *Af, *Bf, *Cf;
        cudaMallocManaged(&Af, sizeof(float) * n * n);
        cudaMallocManaged(&Bf, sizeof(float) * n * n);
        cudaMallocManaged(&Cf, sizeof(float) * n * n);

        std::uniform_real_distribution<float> dist2(-1.0, 1.0);
        for (unsigned int i = 0; i < n * n; i++) {
            Af[i] = dist2(generator);
            Bf[i] = dist2(generator);
        }

        cudaEvent_t startEvent2, stopEvent2;
        cudaEventCreate(&startEvent2);
        cudaEventCreate(&stopEvent2);
        cudaEventRecord(startEvent2, 0);

        matmul_2(Af, Bf, Cf, n, block_dim);

        cudaEventRecord(stopEvent2, 0);
        cudaEventSynchronize(stopEvent2);
        float elapsedTime2;
        cudaEventElapsedTime(&elapsedTime2, startEvent2, stopEvent2);
        cudaEventDestroy(startEvent2);
        cudaEventDestroy(stopEvent2);

        std::cout << Cf[0] << "\n";
        std::cout << Cf[n * n - 1] << "\n";
        std::cout << elapsedTime2 << "\n";

        cudaFree(Af);
        cudaFree(Bf);
        cudaFree(Cf);

        //Double
        double *Ad, *Bd, *Cd;
        cudaMallocManaged(&Ad, sizeof(double) * n * n);
        cudaMallocManaged(&Bd, sizeof(double) * n * n);
        cudaMallocManaged(&Cd, sizeof(double) * n * n);

        std::uniform_real_distribution<double> dist3(-1.0, 1.0);
        for (unsigned int i = 0; i < n * n; i++) {
            Ad[i] = dist3(generator);
            Bd[i] = dist3(generator);
        }

        // kernel call
        cudaEvent_t startEvent3, stopEvent3;
        cudaEventCreate(&startEvent3);
        cudaEventCreate(&stopEvent3);
        cudaEventRecord(startEvent3, 0);

        matmul_3(Ad, Bd, Cd, n, block_dim);

        cudaEventRecord(stopEvent3, 0);
        cudaEventSynchronize(stopEvent3);
        float elapsedTime3;
        cudaEventElapsedTime(&elapsedTime3, startEvent3, stopEvent3);
        cudaEventDestroy(startEvent3);
        cudaEventDestroy(stopEvent3);

        std::cout << Cd[0] << "\n";
        std::cout << Cd[n * n - 1] << "\n";
        std::cout << elapsedTime3 << "\n";

        cudaFree(Ad);
        cudaFree(Bd);
        cudaFree(Cd);
        
    }

    return 0;
}