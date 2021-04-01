#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>
#include <vector>
#include <cmath>

#include "montecarlo.h"
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

int main(int argc, char* argv[]) {
    unsigned int n = std::strtoul(argv[1], nullptr, 10);
    unsigned int t = std::strtoul(argv[2], nullptr, 10);
    float r = 1.0;
    // std::ofstream timefile;
    // timefile.open("timingTask2.txt");

    // for (unsigned int t=1; t<11;t++){
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-r, r);

    std::vector<float> x(n);
    std::vector<float> y(n);
    int incircle=0;
    

    for (size_t i = 0; i < n; i++) {
        x[i] = dist(generator);
        y[i] = dist(generator);
    }
    
    float time = 0.f;
    for (int repeat=0; repeat<10; repeat++){
        start = high_resolution_clock::now();
        omp_set_num_threads(t);     
        incircle = montecarlo(n, x.data(), y.data(), r);
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);   
        time += duration_sec.count();
        float piEst = 4 * (float)incircle / (float)n;
        if (repeat==0){
            std::cout << piEst << '\n';                
        }

    }
    time /= 10;
    std::cout << time << "\n";
    //     timefile << t << " " << time << "\n";
    // }
    
    return 0;
}