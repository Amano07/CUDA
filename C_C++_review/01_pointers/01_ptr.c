#include <stdio.h>

int main(){
    int x = 10;
    int *ptr = &x;
    printf("Adress of x: %p\n", ptr);
    printf("Value of x: %d\n", *ptr);
    
}