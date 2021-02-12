#include <iostream>
#include <random>
#include "scan.h"
#include <chrono>
#include <ratio>
#include <sstream>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]) {
	unsigned int n;
	std::istringstream nn(argv[1]);
	if (nn >> n && nn.eof()) {
		
		float* arr_in = new float[n];
		float* arr_out = new float[n];
		high_resolution_clock::time_point start;
		high_resolution_clock::time_point end;
		duration<double, std::milli> duration_sec;

		int my_seed = 137;
		std::mt19937 generator(my_seed);
		const float min = -1.0, max = 1.0;
		std::uniform_real_distribution<float> dist(min, max);

		for (unsigned int i = 0; i < n; i++) {
			arr_in[i] = dist(generator);
			
		}
		
		// inclusive scan and timing
		start = high_resolution_clock::now();
		scan(arr_in, arr_out, n);
		end = high_resolution_clock::now();
		duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

		cout << duration_sec.count();
		cout << arr_out[0] << "\n";
		cout << arr_out[n - 1] << "\n";

		delete [] arr_in;
		delete [] arr_out;
	}
	
	return 0;
}
