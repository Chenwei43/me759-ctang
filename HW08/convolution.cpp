#include <stddef.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){

    #pragma omp for schedule(static)
	for (size_t k=0; k < n; k++){
		for (size_t l=0; l < n; l++){			
			for (size_t i=0; i<m; i++){
				for (size_t j=0; j<m; j++){
					int idx_fx = (int) k+i-(m-1)/2;
					int idx_fy = (int) l+j-(m-1)/2;

					if (idx_fx >=0 && idx_fx <(int)n && idx_fy>=0 && idx_fy <(int)n){				

						output[k * n + l] += mask[i * m + j] * image[(idx_fx) * n + idx_fy];	
					} else if (idx_fx>=(int)n || idx_fx<0){

						// corner						
						if (idx_fy>=(int)n || idx_fy<0) {
							output[k * n + l] += mask[i * m + j] * 0;
						}
						// edge
						else{
							output[k * n + l] += mask[i * m + j] * 1;
						}														
					}
					else if (idx_fy>=(int)n || idx_fy<0){			
						// can only be edge		
						output[k * n + l] += mask[i * m + j] * 1;							
					} 
				}
			}		
		}
	}
}