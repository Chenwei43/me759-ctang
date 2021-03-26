#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <sstream>

#include "matmul.h"
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

int main(int argc, char* argv[]){
    unsigned int n = std::strtoul(argv[1], nullptr, 10);
    unsigned int t = std::strtoul(argv[2], nullptr, 10);

    // std::ofstream timefile;
    // timefile.open("timingTask1.txt");


    float* A = new float[n*n];
    float* B = new float[n*n];
    float* C = new float[n*n];

    int my_seed = 137;
    std::mt19937 generator(my_seed);
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for (unsigned int i = 0; i < n*n; i++){
        A[i] = dist(generator);
        B[i] = dist(generator);
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    //mmul2
    start = high_resolution_clock::now();
#pragma omp parallel num_threads(t)
    mmul(A, B, C, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << C[0] << "\n";
    cout << C[n*n-1] << "\n";
    cout << duration_sec.count() << "\n";

    // timefile << t << " " << duration_sec.count() << "\n";

    delete [] A;
    delete [] B;
    delete [] C;

    return 0;
}