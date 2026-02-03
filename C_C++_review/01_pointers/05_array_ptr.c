#include <stdio.h>

int main(){
    int arr[] = {12, 24, 36, 48, 60};
    printf("%d\n",arr[0]);
    printf("arr: %p\n", arr);
    int *ptr = arr;

    printf("Position one: %d\n", *ptr);

    for(int i=0; i<5; i++){
        printf("%d\t", *ptr);
        printf("%p\t", ptr);
        printf("%p\n", &ptr);
        ptr++;
    }
}