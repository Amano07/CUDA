#include <stdio.h>
#include <stdlib.h>

int main (){

    int *ptr = NULL;
    printf("1. Initial ptr value: %p\n", (void*)ptr);

    if (ptr == NULL){
        printf("2. ptr is NULL, cannot derefrence\n");
    }

    ptr = malloc(sizeof(int));
    if (ptr == NULL){
        printf("3. Memory allocation failed\n");
        return 1;
    }

    printf("value: %d\n", *(int*)ptr);
}