
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>


#define N 10000000
#define block_size_1D 1024
#define block_size_3D_x 16
#define block_size_3D_y 8
#define block_size_3D_z 8

void vec_add_cpu(float *a, float *b, float *c, int n){
    for (int i=0; i < n ; i++){
        c[i] = a[i] + b[i];
    }
}
__global__ void vec_add_gpu_1D(float *a, float *b, float *c, int n){
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] + b[i];
    }
}

__global__ void vec_add_gpu_3D(float *a, float *b, float *c, int nx, int ny, int nz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz){
        int idx = i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz){
            c[idx] = a[idx] + b[idx];
        } 
    }
}

void init_vector(float *vec, int n){
    for(int i=0; i < n; i++){
        vec[i] = (float)rand() / RAND_MAX;
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1D, *h_c_gpu_3D;
    float *d_a, *d_b, *d_c_1D, *d_c_3D;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu_1D = (float*)malloc(size);
    h_c_gpu_3D = (float*)malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1D, size);
    cudaMalloc(&d_c_3D, size);


    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int num_blocks_1D = (N + block_size_1D - 1) / block_size_1D;

    int nx = 100, ny = 100, nz = 1000;
    dim3 block_size_3D(block_size_3D_x, block_size_3D_y, block_size_3D_z);
    dim3 num_blocks_3D(
        (nx + block_size_3D_x -1) / block_size_3D_x,
        (ny + block_size_3D_y -1) / block_size_3D_y,
        (nz + block_size_3D_z -1) / block_size_3D_z
    );

    printf("Performing warm-up runs...\n");
    for(int i=0; i<3; i++){
        vec_add_cpu(h_a, h_b, h_c_cpu, N);
        vec_add_gpu_1D<<<num_blocks_1D, block_size_1D>>>(d_a, d_b, d_c_1D, N);
        vec_add_gpu_3D<<<num_blocks_3D, block_size_3D>>>(d_a, d_b, d_c_3D, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for(int i=0; i<5; i++){
        double start_time = get_time();
        vec_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double avg_cpu_time = cpu_total_time / 5.0;

    printf("Benchmarking GPU 1D implementation...\n");
    double gpu_total_time_1D = 0.0;
    for(int i=0; i<100; i++){
        double start_time = get_time();
        vec_add_gpu_1D<<<num_blocks_1D, block_size_1D>>>(d_a, d_b, d_c_1D, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time_1D += end_time - start_time;
    }
    double avg_gpu_time_1D = gpu_total_time_1D / 100.0;

    cudaMemcpy(h_c_gpu_1D, d_c_1D, size, cudaMemcpyDeviceToHost);
    bool correct_1D = true;
    for(int i=0; i < N; i++){
        if(fabs(h_c_cpu[i] - h_c_gpu_1D[i]) > 1e-4){
            correct_1D = false;
            std::cout << i << "cpu: " << h_c_cpu[i] << " != " << h_c_gpu_1D[i] << std::endl;
            break;
        }
    }
    printf("1D Results are %s\n", correct_1D ? "correct" : "incorrect");

    printf("Benchmarking GPU 3D implementation...\n");
    double gpu_total_time_3D = 0.0;
    for(int i=0; i<100; i++){
        double start_time = get_time();
        vec_add_gpu_3D<<<num_blocks_3D, block_size_3D>>>(d_a, d_b, d_c_1D, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time_3D += end_time - start_time;
    }
    double avg_gpu_time_3D = gpu_total_time_3D / 100.0;

    cudaMemcpy(h_c_gpu_3D, d_c_3D, size, cudaMemcpyDeviceToHost);
    bool correct_3D = true;
    for(int i=0; i < N; i++){
        if(fabs(h_c_cpu[i] - h_c_gpu_3D[i]) > 1e-4){
            correct_3D = false;
            std::cout << i << "cpu: " << h_c_cpu[i] << " != " << h_c_gpu_3D[i] << std::endl;
            break;
        }
    }
    printf("3D Results are %s\n", correct_3D ? "correct" : "incorrect");

    printf("CPU averge time: %f millisecond \n", avg_cpu_time*1000);
    printf("GPU 1D averge time: %f millisecond \n", avg_gpu_time_1D*1000);
    printf("GPU 3D averge time: %f millisecond \n", avg_gpu_time_3D*1000);

    printf("Speed up: (CPU vs GPU 1D) %fx \n", avg_cpu_time/avg_gpu_time_1D);
    printf("Speed up: (CPU vs GPU 3D) %fx \n", avg_cpu_time/avg_gpu_time_3D);
    printf("Speed up: (GPU 1D vs GPU 3D) %fx \n", avg_gpu_time_1D/avg_gpu_time_3D);


    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1D);
    free(h_c_gpu_3D);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1D);
    cudaFree(d_c_3D);

    return 0;
}