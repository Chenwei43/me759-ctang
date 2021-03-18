#include <stdio.h>
#include<iostream>
#include <omp.h>

void factorial(const unsigned int input, unsigned int fac){
    for (unsigned int j=1; j<input ;j++ ){
        fac *=(j+1);
        
    }
    std::printf("%d!=%d \n", input, fac);
}

int main() {
    unsigned int nThreads = 4;
    std::cout << "Number of threads:" << nThreads << std::endl;
#pragma omp parallel num_threads(nThreads)
    {
        int myId = omp_get_thread_num(); 
        std::printf("I'm thread %d \n", myId);     
    }
#pragma omp parallel for      
    for(unsigned int i=1; i<9 ;i++ ){
        unsigned int fac = 1;
        factorial(i, fac);
    }         
    
    return 0;
}
