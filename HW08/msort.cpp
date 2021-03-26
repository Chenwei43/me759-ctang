#include <omp.h>
#include <stdio.h>
#include <cstring>
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
void merge(int *arr, unsigned int n){

    int *temp = new int[n];

    // start comparing (ascending)
    unsigned int i=0;
    unsigned int j=n/2;
    unsigned int k=0;
    while(i<n/2 && j<n){
        if(arr[i]<=arr[j]){
            temp[k] = arr[i];
            i++;
        }else{
            temp[k] = arr[j];
            j++;
        }
        k++;
    }
    if(i<n/2){
        for(unsigned int res=i; res<n/2; res++){
            temp[k]=arr[res];
            k++;
        }
    }else if(j<n){
        for(unsigned int res=j; res<n; res++){
            temp[k]=arr[res];
            k++;
        }
    }
    memcpy(arr, temp, n*sizeof(int));
    delete []temp;
}


void msort(int* arr, const std::size_t n, const std::size_t threshold){

    if (n<=1) return;
    
    #pragma omp task shared(arr) if (n>threshold)
    {   
        msort(arr, n/2, threshold);
    }
    #pragma omp task shared(arr) if (n>threshold)
    {
        msort(arr+n/2, n/2, threshold);
    }
    #pragma omp taskwait
    {
        merge(arr, n);
    }
        
 

}