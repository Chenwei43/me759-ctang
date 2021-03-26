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

    unsigned int len1 = mid-idL+1;
    unsigned int len2 = idR-mid;
    int temp1[len1], temp2[len2];

    for (unsigned int i=0; i<len1; i++){
        temp1[i] = arr[idL+i];
    }
    for (unsigned int j=0; j<len2; j++){
        temp2[j] = arr[mid + 1+j];
    }

    // start comparing (ascending)
    unsigned int i=0;
    unsigned int j=0;
    unsigned int k=idL;
    while(i<len1 && j<len2){
        if(temp1[i]<=temp2[j]){
            arr[k] = temp1[i];
            i++;
        }else{
            arr[k] = temp2[j];
            j++;
        }
        k++;
    }
    if(i<len1){
        for(unsigned int res=i; res<len1; res++){
            arr[k]=temp1[res];
            k++;
        }
    }else if(j<len2){
        for(unsigned int res=j; res<len2; res++){
            arr[k]=temp2[res];
            k++;
        }
    }
    // while (i < len1) {
    //     arr[k] = temp1[i];
    //     i++;
    //     k++;
    // }
 
    // // Copy the remaining elements of
    // // R[], if there are any
    // while (j < len2) {
    //     arr[k] = temp2[j];
    //     j++;
    //     k++;
    // }

}


void splitSort(int* arr, unsigned int idL, unsigned int idR){
    if (idL>=idR){
        return;
    }
    unsigned int mid = idL+(idR-idL)/2;
    splitSort(arr, idL, mid);
    splitSort(arr, mid+1, idR);
    merge(arr, idL, idR, mid);

}


void msort(int* arr, const std::size_t n, const std::size_t threshold){
    if (n<threshold){
        splitSort(arr, 0, n-1);
    }else{
        // parallel recursive
#pragma omp parallel
    {
        splitSort(arr, 0, n-1);
    }


    }
}