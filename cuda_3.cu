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

#define THREADS_PER_BLOCK 128

using namespace std;

// float distance_cpu(ArffInstance* a, ArffInstance* b) {
//     float sum = 0;
    
//     // iterate through all of the attributes (except the last one, which is the class label)
//     for (int i = 0; i < a->size()-1; i++) {
//         float diff = (a->get(i)->operator float() - b->get(i)->operator float());
//         sum += diff*diff;
//     }
    
//     return sum;
// }


__device__ float distance(float * a, int idx_a, float * b, int idx_b, int size) {
    float sum = 0;

    for (int i = 0; i < size - 1; i++) {
        float diff = * (a + idx_a + i) - * (b +  idx_b + i);
        sum += diff * diff;
    }

    return sum;
}

// help: https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf


__global__ void KNN(float * train, float * test, int * predictions, float * candidates, int * classCounts, int train_size, int test_size, int k, int num_attributes, int num_classes, int stream, int numberElementsPerStream) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    // int queryIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int queryIndex = stream * numberElementsPerStream + tid;

    if (queryIndex < test_size) {
        if (predictions[queryIndex] != INT_MAX){
            return;
        }
        // Stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.

        // access the candidates for the current query item by shifting query_idx times
        // Alternative: candidates array should also be impleted used shared space. TODO: think about it. 
        int candidate_cuda_idx_buffer = queryIndex * 2 * k;

        for (int keyIndex = 0; keyIndex < train_size; keyIndex ++) {

            int train_key_index = keyIndex * num_attributes;
            float dist = distance(test, queryIndex * num_attributes, train, train_key_index, num_attributes);
            // printf("train idx: %d, test idx: %d, dist: %lf\n", keyIndex, queryIndex, dist);

            // Add to our candidates
            for (int c = 0; c < k; c++) {
                if (dist < candidates[candidate_cuda_idx_buffer + 2 * c]) {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for (int x = k - 2; x >= c; x--) {
                        candidates[candidate_cuda_idx_buffer + 2 * x + 2] = candidates[candidate_cuda_idx_buffer + 2 * x];
                        candidates[candidate_cuda_idx_buffer + 2 * x + 3] = candidates[candidate_cuda_idx_buffer + 2 * x + 1];
                    }

                    // Set key vector as potential k NN
                    candidates[candidate_cuda_idx_buffer + 2 * c] = dist;
                    candidates[candidate_cuda_idx_buffer + 2 * c + 1] = train[train_key_index + num_attributes - 1]; // class value

                    break;
                }
            }
        }

        int cuda_idx_buffer = queryIndex * num_classes;

        // Bincount the candidate labels and pick the most common
        for (int i = 0; i < k; i++) {
            classCounts[cuda_idx_buffer + (int) candidates[candidate_cuda_idx_buffer + 2 * i + 1]] += 1;
        }

        int max = -1;
        int max_index = 0;
        for (int i = 0; i < num_classes; i++) {
            if (classCounts[cuda_idx_buffer + i] > max) {
                max = classCounts[cuda_idx_buffer + i];
                max_index = i;
            }
        }

        predictions[queryIndex] = max_index;
        printf("%d %d -- ", queryIndex, max_index);

        // for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }
        // memset(classCounts, 0, num_classes * sizeof(int));
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

    // Allocate and initialize host memory
    float * h_train_instances = (float * ) malloc(train_size * num_attributes * sizeof(float));
    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < num_attributes; j++) {
            h_train_instances[i * num_attributes + j] = train -> get_instance(i) -> get(j) -> operator float();
        }
    }

    float * h_test_instances = (float * ) malloc(test_size * num_attributes * sizeof(float));
    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < num_attributes; j++) {
            h_test_instances[i * num_attributes + j] = test -> get_instance(i) -> get(j) -> operator float();
        }
    }

    // Predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int * h_predictions = (int * ) malloc(test_size * sizeof(int));
    for (int i = 0; i < test_size; i++) {
        h_predictions[i] = INT_MAX;
    }

    // Stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float * h_candidates = (float * ) calloc(test_size * k * 2, sizeof(float));
    for (int i = 0; i < test_size * 2 * k; i++) {
        h_candidates[i] = FLT_MAX;
    }
    // Stores bincounts of each class over the final set of candidate NN
    int * h_class_counts = (int * ) calloc(test_size * num_classes, sizeof(int));

    // Allocate device memory
    float * d_train_instances;
    float * d_test_instances;
    int * d_predictions;
    float * d_candidates;
    int * d_class_counts;

    cudaMalloc( & d_train_instances, train_size * num_attributes * sizeof(float));
    cudaMalloc( & d_test_instances, test_size * num_attributes * sizeof(float));
    cudaMalloc( & d_predictions, test_size * sizeof(int));
    cudaMalloc( & d_candidates, test_size * k * 2 * sizeof(float));
    cudaMalloc( & d_class_counts, test_size * num_classes * sizeof(int));

    // cuda timing: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Copy host memory to device memory
    cudaMemcpy(d_train_instances, h_train_instances, train_size * num_attributes * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_test_instances, h_test_instances, test_size * num_attributes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_predictions, h_predictions, test_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidates, h_candidates, test_size * k * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_class_counts, h_class_counts, test_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);



    int numStreams = 2;
    cudaStream_t *streams = (cudaStream_t*) malloc (numStreams * sizeof(cudaStream_t));

    for (int i = 0; i < numStreams; i++){
        cudaStreamCreate(&streams[i]);
    }


    int numberElementsPerStream = (test_size + numStreams - 1) / numStreams;
    // Configure the block and grid sizes
    int blocksPerGrid = (numberElementsPerStream + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start);
    for (int i = 0; i < numStreams; i++)
    {
        cudaMemcpyAsync(&d_test_instances[i*numberElementsPerStream * num_attributes], &h_test_instances[i*numberElementsPerStream * num_attributes], numberElementsPerStream * num_attributes * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        // cudaMemcpyAsync(&d_predictions[i*numberElementsPerStream], &h_predictions[i*numberElementsPerStream], numberElementsPerStream * sizeof(int), cudaMemcpyHostToDevice, streams[i]);

        KNN <<< blocksPerGrid, THREADS_PER_BLOCK, 0, streams[i] >>> (d_train_instances, d_test_instances, d_predictions, d_candidates, d_class_counts, train_size, test_size, k, num_attributes, num_classes, i, numberElementsPerStream);

        // cudaMemcpyAsync(&h_predictions[i*numberElementsPerStream], &d_predictions[i*numberElementsPerStream], numberElementsPerStream * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();


    // // Launch the kernel function
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
    cudaFree(d_train_instances);
    cudaFree(d_test_instances);
    cudaFree(d_predictions);
    cudaFree(d_candidates);
    cudaFree(d_class_counts);

    // Free host memory
    free(h_train_instances);
    free(h_test_instances);
    free(h_predictions);
    free(h_candidates);
    free(h_class_counts);

    return 0;
}