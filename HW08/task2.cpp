#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <sstream>

#include "convolution.h"
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

int main(int argc, char* argv[]) {
    unsigned int n;
    unsigned int t;
    std::istringstream nn(argv[1]);
    std::istringstream tt(argv[2]);

    // std::ofstream timefile;
    // timefile.open("timingTask2.txt");

    if (nn >> n  && tt >> t && tt.eof()) {
        unsigned int m = 3;
                   
        float* image = new float[n*n];
        float* mask = new float[m*m];
        float* output = new float[n*n];
        
        high_resolution_clock::time_point start;
        high_resolution_clock::time_point end;
        duration<double, std::milli> duration_sec;

        // init w/ random
        int my_seed = 137;
        std::mt19937 generator(my_seed);
        std::uniform_real_distribution<float> dist1(-10.0, 10.0);
        std::uniform_real_distribution<float> dist2(-1.0, 1.0);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++){
                image[i * n + j] = dist1(generator);
                output[i * n + j] = 0;				
            }					
        }
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < m; j++){
                mask[i * m + j] = dist2(generator);	
            }	
        }

        // conv		
        start = high_resolution_clock::now();
#pragma omp parallel num_threads(t)
        convolve(image, output, n, mask, m);
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

        cout << output[0] << "\n";
        cout << output[n*n - 1] << "\n";
        cout << duration_sec.count() << "\n";

        // timefile << t << " " << duration_sec.count() << "\n";

        delete [] image;
        delete [] output;
        delete [] mask;
        


    }

  return 0;
}