#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <tuple>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

// #define THREADS_PER_BLOCK 128

using namespace std;

// https://stackoverflow.com/questions/41050300/how-do-i-allocate-memory-and-copy-2d-arrays-between-cpu-gpu-in-cuda-without-fl

const int array_width = 12;

typedef float my_arr[array_width];

// __global__ void myKernel(my_arr * firstArray, my_arr * secondArray, int rows_train, int rows_test, int columns) {
//     int column = blockIdx.x * blockDim.x + threadIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;

//     if (row >= rows_test || column > 0)
//         return;

    
//     for (int keyIndex = 0; keyIndex < rows_train; keyIndex ++) {
//         float diff_sum = 0;
//         for(int j = 0; j < columns; j++){
//             diff_sum += (firstArray[keyIndex][j] - secondArray[row][j])*(firstArray[keyIndex][j] - secondArray[row][j]);
//         }
//         printf("%d %d %.2f\n", row, column, diff_sum);
//     }


//     // Do something with the arrays like you would on a CPU, like:
//     // firstArray[row][column] = row * 2;
//     // secondArray[row][column] = row * 3;
// }


__global__ void KNN(my_arr * firstArray, my_arr * secondArray, int * predictions, my_arr * candidates, my_arr * classCounts, int train_size, int test_size, int k, int num_attributes, int num_classes) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("%d %d\n", row, column);

    if (row >= test_size || column > 0)
        return;

    for (int keyIndex = 0; keyIndex < train_size; keyIndex ++) {

        float dist = 0;
        for(int j = 0; j < num_attributes - 1; j++){
            float x = (firstArray[keyIndex][j] - secondArray[row][j]);
            dist += x * x;
        }
        // printf("%d %d %.2f\n", row, keyIndex, dist);
        // distance calculation OK

        // TODO: fix rest 
        // seems working

        // Add to our candidates
        for(int c = 0; c < k; c++){
            if(dist < candidates[row][2*c]){
                // Found a new candidate
                // Shift previous candidates down by one
                for(int x = k-2; x >= c; x--) {
                    candidates[row][2*x+2] = candidates[row][2*x];
                    candidates[row][2*x+3] = candidates[row][2*x+1];
                }
                
                // Set key vector as potential k NN
                candidates[row][2*c] = dist;
                candidates[row][2*c+1] = firstArray[keyIndex][num_attributes - 1]; // class value

                // printf("%d %d ", keyIndex, (int)firstArray[keyIndex][num_attributes - 1]);

                break;
            }
        }
    }

    for(int i = 0; i < k;i++){
        classCounts[row][(int)candidates[row][2*i+1]] += 1;
    }
    
    int max = -1;
    int max_index = 0;
    for(int i = 0; i < num_classes;i++){
        if((int) classCounts[row][i] > max){
            max = (int) classCounts[row][i];
            max_index = (int) i;
        }
    }

    predictions[row] = max_index;
    // printf("%d %d -- ", row, predictions[row]);
}


int * computeConfusionMatrix(int * predictions, ArffData * dataset) {
    int * confusionMatrix = (int * ) calloc(dataset -> num_classes() * dataset -> num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses

    for (int i = 0; i < dataset -> num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset -> get_instance(i) -> get(dataset -> num_attributes() - 1) -> operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass * dataset -> num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int * confusionMatrix, ArffData * dataset) {
    int successfulPredictions = 0;

    for (int i = 0; i < dataset -> num_classes(); i++) {
        successfulPredictions += confusionMatrix[i * dataset -> num_classes() + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / (float) dataset -> num_instances();
}




int main(int argc, char * argv[]) {
    if (argc != 4) {
        cout << "Usage: ./main datasets/trainfile.arff datasets/testfile.arff k" << endl;
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData * train = parserTrain.parse();
    ArffData * test = parserTest.parse();

    int num_attributes = train -> num_attributes();
    int num_classes = train -> num_classes();
    int train_size = (int) train -> num_instances();
    int test_size = (int) test -> num_instances();

    // train_size = 60;
    // test_size = 8;

    printf("number of attributes: %d\n", num_attributes);
    printf("number of classes: %d\n", num_classes);
    printf("number of train instances: %d\n", train_size);
    printf("number of test instances: %d\n", test_size);
    
    // printf("train 0, test 0, dist: %.2f\n", distance_cpu(train->get_instance(0), test->get_instance(0)));
    // printf("train 0, test 1, dist: %.2f\n", distance_cpu(train->get_instance(0), test->get_instance(1)));
    // printf("train 0, test 2, dist: %.2f\n", distance_cpu(train->get_instance(0), test->get_instance(2)));
    // printf("train 0, test 3, dist: %.2f\n", distance_cpu(train->get_instance(0), test->get_instance(3)));
    // printf("train 0, test 4, dist: %.2f\n", distance_cpu(train->get_instance(0), test->get_instance(4)));
    // printf("train 0, test 5, dist: %.2f\n", distance_cpu(train->get_instance(0), test->get_instance(5)));
    // printf("train 0, test 6, dist: %.2f\n", distance_cpu(train->get_instance(0), test->get_instance(6)));
    // printf("train 0, test 7, dist: %.2f\n", distance_cpu(train->get_instance(0), test->get_instance(7)));

    // // Allocate and initialize host memory
    // float * h_train_instances = (float * ) malloc(train_size * num_attributes * sizeof(float));
    // for (int i = 0; i < train_size; i++) {
    //     for (int j = 0; j < num_attributes; j++) {
    //         h_train_instances[i * num_attributes + j] = train -> get_instance(i) -> get(j) -> operator float();
    //     }
    // }

    // float * h_test_instances = (float * ) malloc(test_size * num_attributes * sizeof(float));
    // for (int i = 0; i < test_size; i++) {
    //     for (int j = 0; j < num_attributes; j++) {
    //         h_test_instances[i * num_attributes + j] = test -> get_instance(i) -> get(j) -> operator float();
    //     }
    // }

    int rows_train = train_size, rows_test = test_size, columns = array_width;
    my_arr * h_firstArray, * h_secondArray, *h_candidates, *h_class_counts;
    my_arr * d_firstArray, * d_secondArray, *d_candidates, *d_class_counts;
    size_t dsize_train = rows_train * columns * sizeof(float);
    size_t dsize_test = rows_test * columns * sizeof(float);
    size_t dsize_candidates = rows_test * columns * sizeof(float);
    size_t dsize_class_count = rows_test * columns * sizeof(float);
    h_firstArray = (my_arr * ) malloc(dsize_train);
    h_secondArray = (my_arr * ) malloc(dsize_test);
    h_candidates = (my_arr * ) malloc(dsize_candidates);
    h_class_counts = (my_arr * ) malloc(dsize_class_count);

    // populate h_ arrays
    memset(h_firstArray, 0, dsize_train);
    memset(h_secondArray, 0, dsize_test);
    memset(h_candidates, 0, dsize_candidates);
    memset(h_class_counts, 0, dsize_class_count);

    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < num_attributes; j++) {
            h_firstArray[i][j] = train -> get_instance(i) -> get(j) -> operator float();
        }
        // printf("%d .. ", (int) h_firstArray[i][num_attributes - 1]);
    }

    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < num_attributes; j++) {
            h_secondArray[i][j] = test -> get_instance(i) -> get(j) -> operator float();
        }
    }

    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < columns; j++) {
            h_candidates[i][j] = FLT_MAX;
        }
    }

    // Allocate memory on device
    cudaMalloc( & d_firstArray, dsize_train);
    cudaMalloc( & d_secondArray, dsize_test);
    cudaMalloc( & d_candidates, dsize_candidates);
    cudaMalloc( & d_class_counts, dsize_class_count);
    // Do memcopies to GPU
    cudaMemcpy(d_firstArray, h_firstArray, dsize_train, cudaMemcpyHostToDevice);
    cudaMemcpy(d_secondArray, h_secondArray, dsize_test, cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidates, h_candidates, dsize_candidates, cudaMemcpyHostToDevice);
    cudaMemcpy(d_class_counts, h_class_counts, dsize_class_count, cudaMemcpyHostToDevice);



    // Predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int * h_predictions = (int * ) malloc(test_size * sizeof(int));

    // Stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    // float * h_candidates = (float * ) calloc(test_size * k * 2, sizeof(float));
    // for (int i = 0; i < test_size * 2 * k; i++) {
        // h_candidates[i] = FLT_MAX;
    // }
    // Stores bincounts of each class over the final set of candidate NN
    // int * h_class_counts = (int * ) calloc(test_size * num_classes, sizeof(int));

    // Allocate device memory
    // float * d_train_instances;
    // float * d_test_instances;
    int * d_predictions;
    // float * d_candidates;
    // int * d_class_counts;

    // cudaMalloc( & d_train_instances, train_size * num_attributes * sizeof(float));
    // cudaMalloc( & d_test_instances, test_size * num_attributes * sizeof(float));
    cudaMalloc( & d_predictions, test_size * sizeof(int));
    // cudaMalloc( & d_candidates, test_size * k * 2 * sizeof(float));
    // cudaMalloc( & d_class_counts, test_size * num_classes * sizeof(int));

    // cuda timing: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Copy host memory to device memory
    // cudaMemcpy(d_train_instances, h_train_instances, train_size * num_attributes * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_test_instances, h_test_instances, test_size * num_attributes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_predictions, h_predictions, test_size * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_candidates, h_candidates, test_size * k * 2 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_class_counts, h_class_counts, test_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the block and grid sizes
    // int blocksPerGrid = (test_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    dim3 block(16, 8);
    dim3 grid((columns + block.x - 1) / block.x, (rows_train + block.y - 1) / block.y);

    cudaEventRecord(start);

    KNN <<< grid, block >>> (d_firstArray, d_secondArray, d_predictions, d_candidates, d_class_counts, train_size, test_size, k, num_attributes, num_classes);

    // Launch the kernel function
    // KNN <<< blocksPerGrid, THREADS_PER_BLOCK >>> (d_train_instances, d_test_instances, d_predictions, d_candidates, d_class_counts, train_size, test_size, k, num_attributes, num_classes);

    // Transfer device results to host memory
    cudaMemcpy(h_predictions, d_predictions, test_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t cudaError = cudaGetLastError();

    if (cudaError != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Compute the confusion matrix
    int * confusionMatrix = computeConfusionMatrix(h_predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    printf("The %i-NN classifier for %d test instances on %d train instances required %f ms CPU time. Accuracy was %.2f%%\n", k, test_size, train_size, milliseconds, (accuracy * 100));

    // Free device global memory
    cudaFree(d_firstArray);
    cudaFree(d_secondArray);
    cudaFree(d_predictions);
    cudaFree(d_candidates);
    cudaFree(d_class_counts);

    // Free host memory
    free(h_firstArray);
    free(h_secondArray);
    free(h_predictions);
    free(h_candidates);
    free(h_class_counts);

    return 0;
}
