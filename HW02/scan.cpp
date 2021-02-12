#include <stddef.h>
#include <stdlib.h>
#include "scan.h"

void scan(const float* arr, float* output, std::size_t n) {
	for (std::size_t i = 0; i < n; i++) {
		output[i] = arr[i];
		if (i>0){
			output[i] += output[i-1];
		}
		
	}
}