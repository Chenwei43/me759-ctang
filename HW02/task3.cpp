#include <iostream>
#include <random>
#include <chrono>
#include <sstream>

#include "matmul.h"
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

int main(){
  unsigned int n = 1500;
  cout << n << "\n";  
  double* A = new double[n*n];
  double* B = new double[n*n];
  double* C = new double[n*n];

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


  //mmul1
  start = high_resolution_clock::now();
  mmul1(A, B, C, n);
  end = high_resolution_clock::now();
  duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
  cout << duration_sec.count() << "\n";
  cout << C[n*n-1] << "\n";

  //mmul2
  start = high_resolution_clock::now();
  mmul2(A, B, C, n);
  end = high_resolution_clock::now();
  duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
  cout << duration_sec.count() << "\n";
  cout << C[n*n-1] << "\n";

  //mmul3
  start = high_resolution_clock::now();
  mmul3(A, B, C, n);
  end = high_resolution_clock::now();
  duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
  cout << duration_sec.count() << "\n";
  cout << C[n*n-1] << "\n";

  //mmul4, use the same values for A and B, just different type.
  std::vector<double> A4(n*n);
  std::vector<double> B4(n*n);
  for (size_t h = 0; h < n*n; h++){
    A4[h] = A[h];
    B4[h] = B[h];
  }

  start = high_resolution_clock::now();
  mmul4(A4, B4, C, n);
  end = high_resolution_clock::now();
  duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
  cout << duration_sec.count() << "\n";
  cout << C[n*n-1] << "\n";

  delete [] A;
  delete [] B;
  delete [] C;

  return 0;
}