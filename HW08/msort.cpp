#include <omp.h>
#include "msort.h"


// This function does a merge sort on the input array "arr" of length n. 
// You can add more functions as needed to complete the merge sort,
// but do not change this file. Declare and define your addtional
// functions in the msort.cpp file, but the calls to your addtional functions
// should be wrapped in the "msort" function.

// "threshold" is the lower limit of array size where your function would 
// start making parallel recursive calls. If the size of array goes below
// the threshold, a serial sort algorithm will be used to avoid overhead
// of task scheduling
void merge(int *arr, unsigned int idL, unsigned int idR, unsigned int mid){
    
}


void splitSort(int* arr, unsigned int idL, unsigned int idR){
    if (idL>=idR){
        return;
    }
    unsigned int mid = idL+(idR-1)/2;
    splitSort(arr, idL, mid);
    splitSort(arr, mid+1, idR);
    merge(arr, idL, idR, mid);

}

void msort(int* arr, const std::size_t n, const std::size_t threshold){

}