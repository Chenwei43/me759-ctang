#include <iostream>
#include <random>
#include <chrono>
#include <sstream>

#include "convolution.h"
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

int main(int argc, char* argv[]) {
  unsigned int n, m;
	std::istringstream nn(argv[1]);
  std::istringstream mm(argv[2]);

  if (nn >> n && mm >> m && mm.eof()) {
		float* image = new float[n*n];
		float* mask = new float[m*m];
		float* image_out = new float[n*n];
        
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
			}					
		}
    for (size_t i = 0; i < m; i++) {
			for (size_t j = 0; j < m; j++){
        mask[i * m + j] = dist2(generator);	
      }	
		}

		// conv		
		start = high_resolution_clock::now();
		convolve(image, image_out, n, mask, m);
		end = high_resolution_clock::now();
		duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

		cout << duration_sec.count() << "\n";
		cout << image_out[0] << "\n";
		cout << image_out[n*n - 1] << "\n";

		delete [] image;
		delete [] image_out;
		delete [] mask;
	
  }

  return 0;
}