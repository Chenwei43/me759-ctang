#include <stdio.h>
#include <stdlib.h>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    exit(1);
  }
  int arg1 = atoi(argv[1]); 

  for (int i = 0; i <= arg1; i++){
    printf("%d ", i);
    if (i == arg1){
      printf("\n");
    }
  }
  
  for (int i = arg1; i >=0; i--){
    std::cout << i << " ";
    if (i == 0){
      printf("\n");
    }
  }

  return 0;
}
