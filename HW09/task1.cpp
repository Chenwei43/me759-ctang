#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>
#include <vector>
#include <cmath>

#include "cluster.h"
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

int main(int argc, char* argv[]) {
    unsigned int n = std::strtoul(argv[1], nullptr, 10);
    unsigned int t = std::strtoul(argv[2], nullptr, 10);

    // std::ofstream timefile;
    // timefile.open("timingTask1.txt");

    // for (t=1; t<11;t++){
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_int_distribution<int> dist(0, n);

    std::vector<int> arr(n);
    std::vector<int> centers(t);
    std::vector<int> dists(t, 0);

    for (size_t i = 0; i < n; i++) {
        arr[i] = dist(generator);
        //arr[i] = i;
    }
    std::sort(arr.begin(), arr.end());
    for (size_t i = 1; i < t+1; i++) {
        centers[i-1] = (2*i-1)*n / (2*t);
    }
    float time = 0.f;
    for (int repeat=0; repeat<10; repeat++){
        start = high_resolution_clock::now();
        cluster(n, t, arr.data(), centers.data(), dists.data());
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);        
        std::vector<int>::iterator dist_max = std::max_element(dists.begin(), dists.end());
        time += duration_sec.count();
        if (repeat==0){
            std::cout << dist_max[0] << '\n';
            std::cout << std::distance(dists.begin(), dist_max) << '\n';
            //std::cout << duration_sec.count() << "\n";
        }

    }
    time /= 10;
    std::cout << time << "\n";
        // timefile << t << " " << time << "\n";
    // }

    return 0;
}