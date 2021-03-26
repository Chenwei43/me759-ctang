#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <sstream>

#include "msort.h"
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

int main(int argc, char* argv[]){
    unsigned int n = std::strtoul(argv[1], nullptr, 10);
    unsigned int ts = std::strtoul(argv[2], nullptr, 10);

    std::ofstream timefile;
    timefile.open("timingTask3.txt");

    for (unsigned int t=8; t<9; t++){
        int* arr = new int[n];
        int my_seed = 137;
        std::mt19937 generator(my_seed);
        std::uniform_int_distribution<int> dist1(-1000, 1000);
        for (size_t i = 0; i < n; i++) {            
            arr[i] = dist1(generator);	        					
        }
        high_resolution_clock::time_point start;
        high_resolution_clock::time_point end;
        duration<double, std::milli> duration_sec;
        start = high_resolution_clock::now();
        omp_set_num_threads(t);
        msort(arr, n, ts);
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

        cout << arr[0] << "\n";
        cout << arr[n - 1] << "\n";
        cout << duration_sec.count() << "\n";

        timefile << t << " " << duration_sec.count() << "\n";
    }

    return 0;

}

