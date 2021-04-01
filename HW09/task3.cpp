#include "mpi.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>
#include <vector>
#include <cmath>
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

int main(int argc, char* argv[]) {
    unsigned int n = std::strtoul(argv[1], nullptr, 10);
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> t0;
    duration<double, std::milli> t1;

        
    float* buffer1 = new float[n];
    float* buffer2 = new float[n];

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_int_distribution<int> dist(0, n);
    for (size_t j = 0; j < n; j++) {
        buffer1[j] = dist(generator);
        buffer2[j] = dist(generator);
    }
    int         rank;       /* rank of process      */
    int         p;             /* number of processes  */
    int         tag = 0;       /* tag for messages     */
    float       t;
    MPI_Status  status;        /* return status for receive  */

    MPI_Init(&argc, &argv); // Start up MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Find out process rank   
    MPI_Comm_size(MPI_COMM_WORLD, &p); // Find out number of processes

    if (rank == 0) {
        start = high_resolution_clock::now();
        MPI_Send(buffer1, n, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
        //printf("rank = %d, sent %f\n", rank, buffer1[0]);
        MPI_Recv(buffer2, n, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, &status);
        //printf("rank = %d, received %f\n", rank, buffer2[0]);

        end = high_resolution_clock::now();
        t0 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);   
        t = t0.count();
        //printf("rank = %d, t0 = %f\n", rank, t);
        MPI_Send(&t, 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);

        
    } 
    else if (rank == 1){   
        start = high_resolution_clock::now();
        MPI_Recv(buffer1, n, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
        //printf("rank = %d, received %f\n", rank, buffer1[0]);                     
        MPI_Send(buffer2, n, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
        //printf("rank = %d, sent %f\n", rank, buffer2[0]);
        end = high_resolution_clock::now();
        t1 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        // float t11 = t1.count();   
        // printf("rank = %d, t1 = %f\n", rank, t11);
        MPI_Recv(&t, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
        //printf("rank = %d, recv, t0 = %f\n", rank, t);
        t += t1.count();
        printf("%f\n", t);
        
    }

    MPI_Finalize(); // Shut down MPI
    return 0;
}
    