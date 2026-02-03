#include <iostream>

using namespace std;

typedef struct {
    float x;
    float y;
} Point;

int main(){

    Point p = {1.1, 2.2};
    //printf("size of Point p: %zu\n", sizeof(Point));
    cout << "size of Point p:" << sizeof(Point) << endl;
}
