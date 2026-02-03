#include <stdio.h>

int main(){
    int arr[] = {12, 24, 36, 48, 60}; 
    
    size_t size = sizeof(arr) / sizeof(arr[0]); // Output = 5
    // sizeof(arr) -> 5 * 4 = 20
    // sizeof(arr[0]) -> 4
    printf("Size of arr: %zu\n", size);
    printf("size of size_t: %zu\n", sizeof(size_t));
    printf("int size in bytes: %zu\n", sizeof(int));
    // z -> size_t;
    // u -> unsigned int
    // %zu -> size_t

    return 0;
}