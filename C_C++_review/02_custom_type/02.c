#include <stdio.h>

typedef struct {
    float x;
    float y;
} Point;

int main(){

    Point p = {1.1, 2.2};
    printf("size of Point p: %zu\n", sizeof(Point));
}

