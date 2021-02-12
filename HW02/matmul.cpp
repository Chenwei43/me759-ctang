#include <stddef.h>
#include <stdlib.h>
#include "matmul.h"

void mmul1(const double* A, const double* B, double* C, const unsigned int n){

  for (size_t i=0; i<n; i++){
    for (size_t j=0; j<n; j++){
      C[i*n+j] = 0;
      for (size_t k=0; k<n; k++){
        C[i*n+j] += A[i*n+k]*B[j+k*n];
      }
    }
  }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n){
  for (size_t i=0; i<n; i++){
    for (size_t k=0; k<n; k++){
      C[k*n+i] = 0;
      for (size_t j=0; j<n; j++){
        C[k*n+i] += A[k*n+j]*B[i+j*n];
      }
    }
  }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n){
  for(size_t j=0; j<n; j++){
    for(size_t k=0; k<n; k++){
      C[j*n+k] = 0;
      for(size_t i=0; i<n; i++){
        C[j*n+k] += A[i+j*n]*B[k+i*n];
      }
    }
  }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n){
 for (size_t i=0; i<n; i++){
    for (size_t j=0; j<n; j++){
      C[i*n+j] = 0;
      for (size_t k=0; k<n; k++){
        C[i*n+j] += A[i*n+k]*B[j+k*n];
      }
    }
  }

}