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

#define THREADS_PER_BLOCK 16

using namespace std;


// help: https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf

__device__ float distance(float * a, int idx_a, int idx_b, int size) {
    float sum = 0;

    for (int i = 0; i < size - 1; i++) {
        float diff = * (a + idx_a + i) - * (a +  idx_b + i);
        sum += diff * diff;
    }

    return sum;
}


__global__ void KNN(float * test, int * predictions, int train_size, int test_size, int k, int num_attributes, int num_classes) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row    = blockIdx.y * blockDim.y + threadIdx.y;
    int queryIndex = row * THREADS_PER_BLOCK + column;
    // printf("%d %d %d\n", row, column, queryIndex);
    // int queryIndex = stream * numberElementsPerStream + tid;

    if (queryIndex < test_size) {
        // printf("%d %d %d\n", row, column, queryIndex);

        // Stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
        float candidates[11];
        // Stores bincounts of each class over the final set of candidate NN
        int class_counts[11];

        for (int i = 0; i < 11; i++){
            candidates[i] = FLT_MAX;
            class_counts[i] = 0;
        }

        __syncthreads();

        for (int keyIndex = 0; keyIndex < train_size; keyIndex ++) {
            float dist = distance(test, queryIndex * num_attributes, num_attributes * test_size + keyIndex * num_attributes, num_attributes);
            __syncthreads();


            // Add to our candidates
            for (int c = 0; c < k; c++) {
                if (dist < candidates[2 * c]) {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for (int x = k - 2; x >= c; x--) {
                        candidates[2 * x + 2] = candidates[2 * x];
                        candidates[2 * x + 3] = candidates[2 * x + 1];
                    }
                    // Set key vector as potential k NN
                    candidates[2 * c] = dist;
                    candidates[2 * c + 1] = test[num_attributes * test_size + keyIndex * num_attributes + num_attributes - 1]; // class value
                    break;
                }
            }
        }
        __syncthreads();


        // Bincount the candidate labels and pick the most common
        for (int i = 0; i < k; i++) {
            class_counts[ (int) candidates[2 * i + 1]] ++;
        }

        int max = -1;
        int max_index = 0;
        for (int i = 0; i < num_classes; i++) {
            if (class_counts[ i] > max) {
                max = class_counts[i];
                max_index = i;
            }
        }
        predictions[queryIndex] = max_index;
    }
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

    printf("number of attributes: %d\n", num_attributes);
    printf("number of classes: %d\n", num_classes);
    printf("number of train instances: %d\n", train_size);
    printf("number of test instances: %d\n", test_size);


    float * h_test_instances = (float * ) malloc((train_size + test_size) * num_attributes * sizeof(float));
    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < num_attributes; j++) {
            h_test_instances[i * num_attributes + j] = test -> get_instance(i) -> get(j) -> operator float();
        }
    }
    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < num_attributes; j++) {
            h_test_instances[ (test_size + i) * num_attributes + j] = train -> get_instance(i) -> get(j) -> operator float();
        }
    }

    // Predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int * h_predictions = (int * ) malloc(test_size * sizeof(int));


    // Allocate device memory
    float * d_test_instances;
    int * d_predictions;

    cudaMalloc( & d_test_instances, (test_size + train_size) * num_attributes * sizeof(float));
    cudaMalloc( & d_predictions, test_size * sizeof(int));

    // cuda timing: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Copy host memory to device memory
    cudaMemcpy(d_test_instances, h_test_instances, (train_size + test_size) * num_attributes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_predictions, h_predictions, test_size * sizeof(int), cudaMemcpyHostToDevice);

    // Configure the block and grid sizes
    int gridDim = (test_size + THREADS_PER_BLOCK - 1) / (THREADS_PER_BLOCK);

    dim3 blockSize (THREADS_PER_BLOCK, 1);
    dim3 gridSize (gridDim, 1);

    cudaEventRecord(start);
    KNN <<< gridSize, blockSize >>> (d_test_instances, d_predictions, train_size, test_size, k, num_attributes, num_classes);
    cudaDeviceSynchronize();                                                                                                                                                    

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
    cudaFree(d_test_instances);
    cudaFree(d_predictions);

    // Free host memory
    free(h_test_instances);
    free(h_predictions);

    return 0;
}