/* 
 * Course: High Performance Computing 2023/2024
 * 
 * Lecturer: Moscato Francesco	fmoscato@unisa.it
 * 
 * Student: Ferraioli Chiara	0622702169	c.ferraioli30@studenti.unisa.it
 * 
 * 									REQUIREMENTS OF ASSIGNMENT
 * Student shall provide a parallel version of the Prim's algorithm to find the minimum spanning tree of
 * a connected, non-oriented and weighted graph with both "OpenMP + MPI" and "OpenMP + Cuda" approaches, 
 * comparing results with a known solution on single-processing node. Results and differences shall be 
 * discussed for different inputs (type and size). The parallel algorithm used in "OpenMP + MPI" solution 
 * could not be the same of the "OpenMP + CUDA" approach.
 * 
 * This source file is the parallel implementation of the Prim's algorithm to find the minimum spanning
 * tree of a connected, non-oriented and weighted graph, realized using the "CUDA+OpenMP" approach.
 * 
 * The program takes five command-line arguments: the number of omp threads, the number of vertices, 
 * graph density, optimization level and number of file.
 * 
 * Usage: Cuda_prim [numberOfThreads] [numberOfVertices] [graphDensity] [optimizationLevel] [fileNumber]
 * 
 * Copyright (C) 2024 - All Rights Reserved
 * 
 * This file is part of ProjectHPC
 * 
 * ProjectHPC is free software: you can redistribute it and/or modify it under the terms of 
 * the GNU General Public License as published by the Free Software Foundation, either version 
 * 3 of the License, or (at your option) any later version.
 *
 * ProjectHPC is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ProjectHPC. 
 * If not, see <http://www.gnu.org/licenses/>.
 * 
 *
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <omp.h>
#include <cooperative_groups.h>
#include "../Headers/TMatrix.h"
#include "../Headers/PStructs.h"
#include "../Headers/graphGenerator.h"

// Define for checking CUDA errors
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){

    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


namespace cg = cooperative_groups;
#define FULL_MASK 0xffffffff

/**
 * @brief Reduces the minimum edge within a warp.
 *
 * @param edge The edge structure containing src, dest, and weight.
 * @return The minimum edge within the warp.
 */
__inline__ __device__ TEdge warpReduceMin(TEdge edge){

    //Uses the shuffle function to exchange the weight, src, and dest values among threads within the warp.
    for (int offset = warpSize / 2; offset > 0; offset /= 2){
        int tmpWeight = __shfl_down_sync(FULL_MASK, edge.weight, offset);
        int src = __shfl_down_sync(FULL_MASK, edge.src, offset);
        int dest = __shfl_down_sync(FULL_MASK, edge.dest, offset);

        /*Checks if the temporary weight (tmpWeight) is less than the weight of the current edge (edge).
        If so, updates the weight and the other values of the edge with the temporary values.*/
        if(tmpWeight < edge.weight){
            edge.weight = tmpWeight;
            edge.src = src;
            edge.dest = dest;
        }
    }
    return edge;
}

/**
 * @brief Reduces the minimum edge within a block.
 *
 * @param edge The edge structure containing src, dest, and weight.
 * @return The minimum edge within the block.
 */
__inline__ __device__ TEdge blockReduceMin(TEdge edge){

    // Shared memory to store intermediate results within a block.
    static __shared__ TEdge shared[32];

    // Determine the thread's lane (position within the warp) and warp identifier.
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Initialize a temporary edge with maximum weight.
    TEdge tmpEdge = {-1, -1, INT_MAX};

    // Reduce the weight within the warp using the warpReduceMin function.
    edge = warpReduceMin(edge);

    // Store the result of the reduction in shared memory at the warp identifier's position.
    if(lane == 0){
        shared[wid] = edge;
    }

    // Synchronize to ensure all threads have stored their values in shared memory.
    __syncthreads();

    /* Determine if the thread is within the first warp of the block.
    If so, assign the edge value from shared memory; otherwise, use the temporary edge.*/
    edge = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : tmpEdge;

    // If the thread is in the first warp, further reduce the edge values to find the minimum.
    if(wid == 0){
        edge = warpReduceMin(edge);
    }

    return edge;

}

/**
 * @brief Searches for the minimum edge for a specific group of vertices assigned to the block.
 *
 * @param graph The input graph in adiacency matrix representation.
 * @param minEdges An array of minimum edges for each vertex assigned to the block.
 * @param min The minimum edge found.
 * @param inMST A boolean array representing whether a vertex is included in MST.
 * @param row The last vertex added to the MST.
 */
__inline__ __device__ void minEdgeSearch(TMatrix *graph, TEdge *minEdges, TEdge *min, bool *inMST, int row){
    int i;
    int element;
    int V = graph->cols;
    
    // Initialize a structure to store the minimum weight edge found.
    TEdge localMin = {-1, -1, INT_MAX};

    // Iterate over the vertices within the block's assigned range.
    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += blockDim.x * gridDim.x){
        
        // Check if the vertex is not yet included in the MST.
        if(!inMST[i]){
            element = graph->items[(row * V) + i];

            /* If an edge exists and has a lower weight than the current minimum for the vertex, 
            update the minimum.*/
            if(element != 0 && element < minEdges[i].weight){
                minEdges[i].dest = row;
                minEdges[i].weight = element;
            }

            // Compare the weight of the current edge to the minimum weight found within this block.
            if(minEdges[i].weight < localMin.weight){
                localMin = minEdges[i];
            }
        }
    }

    // Reduce the minimum weight edge within the block.
    localMin = blockReduceMin(localMin);

    // If the thread is the first in its block, update the minimum weight edge for the entire block.
    if(threadIdx.x == 0){
        int oldWeight = atomicMin(&(min->weight), localMin.weight);

        // If the updated minimum weight matches the block's minimum, update the source and destination.
        if(localMin.weight < oldWeight){
            min->src = localMin.src;
            min->dest = localMin.dest;
        }
    }
}

/**
 * @brief CUDA kernel to perform Prim's algorithm.
 *
 * @param graph The input graph in adiacency matrix representation.
 * @param inMST A boolean array representing whether a vertex is included in MST.
 * @param minEdges An array of minimum edges for each vertex.
 * @param MST The resulting Minimum Spanning Tree.
 * @param min The minimum edge found.
 */
__global__ void primMST(TMatrix *graph, bool *inMST, TEdge *minEdges, TEdge *MST, TEdge *min){
    int i;
    int row;
    int V = graph->cols;
    
    // Create a grid group for block synchronization.
    cg::grid_group grid = cg::this_grid();

    // Initialize the starting vertex.
    row = 0;

    // Iterate to find V - 1 edges for the MST.
    for(i = 0; i < V - 1; i++){

        // Reset the minimum weight for each iteration.
        min->weight = INT_MAX;

        // Search for the minimum weight edge to include in the MST for the current row.
        minEdgeSearch(graph, minEdges, min, inMST, row);

        // Synchronize the grid to ensure all blocks finish searching.
        grid.sync();

        // If this is the first thread in the first block of the grid, update the MST with the minimum weight edge.
        if(blockDim.x * blockIdx.x + threadIdx.x == 0){
            MST[min->src].src = min->src;
            MST[min->src].dest = min->dest;
            MST[min->src].weight = min->weight;
            // Mark the selected vertex as included in the MST.
            inMST[min->src] = true;
        }

        // Update the current row to the source of the minimum weight edge.
        row = min->src;

        // Synchronize the grid again for the next iteration.
        grid.sync();
    }
}

/**
 * @brief Allocates memory for variables used in the algorithm.
 *
 * @param MST The Minimum Spanning Tree.
 * @param inMST A boolean array representing whether a vertex is included in MST.
 * @param minEdges An array of minimum edges for each vertex.
 * @param d_graph The device graph data structure.
 * @param d_minEdges The device array of minimum edges for each vertex.
 * @param d_MST The device Minimum Spanning Tree.
 * @param d_inMST The device boolean array representing whether a vertex is included in MST.
 * @param d_items The device array for graph data.
 * @param min The minimum edge found.
 * @param V The number of vertices in the graph.
 */
void allocateVariables(TEdge **MST, bool **inMST, TEdge **minEdges, TMatrix ** d_graph, TEdge **d_minEdges,
        TEdge **d_MST, bool **d_inMST, int **d_items, TEdge **min, int V){

    *MST = (TEdge*) malloc (V * sizeof(TEdge));
    *inMST = (bool*) malloc (V * sizeof(bool));
    *minEdges = (TEdge*) malloc (V * sizeof(TEdge));

    CUDA_CHECK(cudaMalloc(d_graph, sizeof(TMatrix)));
    CUDA_CHECK(cudaMalloc(d_items, V * V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(d_inMST,  V * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(d_minEdges, V * sizeof(TEdge)));
    CUDA_CHECK(cudaMalloc(d_MST, V * sizeof(TEdge)));
    CUDA_CHECK(cudaMalloc(min, sizeof(TEdge)));

}

/**
 * @brief Rounds down an integer to the nearest power of two.
 *
 * This function finds the largest power of two that is less than or equal to the input value.
 * It's used to ensure that the number of threads per block is a power of two, which is often
 * beneficial for performance due to the way warp scheduling works on CUDA architectures.
 *
 * @param x The integer to be rounded down.
 * @return The largest power of two less than or equal to x.
 */
int roundDownToPowerOfTwo(int x) {
    int power = 1;
    while (power * 2 <= x) {
        power *= 2;
    }
    return power;
}

/**
 * @brief Computes the optimal block size for the kernel execution on the GPU.
 *
 * This function calculates the optimal number of threads per block and the size of the grid
 * for the GPU kernel, based on the device's capabilities and the kernel's requirements.
 * It ensures that the number of threads per block is a power of two and that the grid size does not
 * exceed the device's maximum grid size limits. The function uses CUDA runtime API calls to
 * query the device properties and to calculate occupancy.
 *
 * @param V The number of elements or tasks to be processed by the kernel.
 * @param gridSize Pointer to the dim3 structure where the computed grid size will be stored.
 * @param blockSize Pointer to the dim3 structure where the computed block size will be stored.
 */
void computeBlockSize(int V, dim3* gridSize, dim3* blockSize){
    int minGridSize; // Minimum grid size needed to achieve maximum occupancy.
    int blockSizeToUse; // Optimal block size calculated based on kernel and device properties.
    int maxThreadsPerBlock, maxGridSizeX; // Device properties: max threads per block and max grid dimensions.

    // Get the maximum number of threads per block supported by the device.
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));

    // Get the maximum size of the grid in the x-dimension.
    CUDA_CHECK(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, 0));

    // Calculate the optimal block size for maximum occupancy.
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeToUse, (void*)primMST, 0, V));

    // Round the block size down to the nearest power of two.
    blockSizeToUse = roundDownToPowerOfTwo(blockSizeToUse);

    // Ensure the block size does not exceed the device's maximum capability.
    blockSizeToUse = min(blockSizeToUse, maxThreadsPerBlock);

    // Set the calculated block size.
    blockSize->x = blockSizeToUse;

    // Calculate the required grid size, making sure it does not exceed the device's limits.
    int requiredGridSize = (V + blockSizeToUse - 1) / blockSizeToUse;
    gridSize->x = min(requiredGridSize, maxGridSizeX);
}

/**
 * @brief Main function to run the Prim's algorithm using CUDA and OpenMP.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return int Return code.
 */
int main(int argc, char *argv[]){

    // Check if the correct number of command-line arguments is provided.
    if(argc != 6){
        fprintf(stderr, "Usage:\n\t%s [numberOfThreads] [numberOfVertices] [graphDensity] [optimizationLevel] [fileNumber]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Check for valid input arguments.
    if(atoi(argv[1]) <= 0){
        fprintf(stderr, "Invalid number of threads!\n");
        exit(EXIT_FAILURE);
    }

    if(atoi(argv[2]) <= 0){
        fprintf(stderr, "Invalid number of vertices!\n");
        exit(EXIT_FAILURE);
    }

    if(atof(argv[3]) <= 0.0 || atof(argv[3]) >= 1.0){
        fprintf(stderr, "Invalid graph density! It must be a number between 0 and 1!\n");
        exit(EXIT_FAILURE);
    }

    if(atoi(argv[4]) < 0 && atoi(argv[4]) > 3){
        fprintf(stderr, "Invalid optimization! It must be a number between 0 and 3!\n");
        exit(EXIT_FAILURE);
    }

    if(atoi(argv[5]) < 0){
        fprintf(stderr, "Invalid optimization! It must be a number between 0 and 3!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize variable to measure execution time
    struct timeval total_start;
    gettimeofday(&total_start, NULL);

    // Initialize variables and allocate memory for CUDA data structures.
    dim3 gridSize;
    dim3 blockSize;
    int V;
    int i;
    int opt;
    int fileNumber;
    double density;
    TMatrix* graph;
    TMatrix* d_graph;
    int* d_items;
    TEdge* minEdges;
    TEdge* d_minEdges;
    TEdge* MST;
    TEdge* d_MST;
    TEdge* min;
    bool* inMST;
    bool* d_inMST;

    int num_threads = atoi(argv[1]);
    V = atoi(argv[2]);
    density = atof(argv[3]);
    opt = atoi(argv[4]);
    fileNumber = atoi(argv[5]);

    // Set the number of OpenMP threads.
    omp_set_num_threads(num_threads);

    // Allocate memory for data structures.
    allocateVariables(&MST, &inMST, &minEdges, &d_graph, &d_minEdges, &d_MST, &d_inMST, &d_items, &min, V);

    struct timeval omp_start;
    gettimeofday(&omp_start, NULL);
    // Initialize inMST and minEdges arrays.
    #pragma omp parallel for default(none) shared(V, inMST, minEdges) private(i)
    for(i = 1; i < V; i++){
        inMST[i] = false;
        minEdges[i].src = i;
        minEdges[i].dest = -1;
        minEdges[i].weight = INT_MAX;
    }
    inMST[0] = true;

    struct timeval omp_end;
    gettimeofday(&omp_end, NULL);
    double total_omp = ((double)omp_end.tv_sec + (double)omp_end.tv_usec / 1000000.0) - ((double)omp_start.tv_sec + (double)omp_start.tv_usec / 1000000.0);

    // Generate a random connected, non-oriented and weighted graph.
    struct timeval creation_start;
    gettimeofday(&creation_start, NULL);
    graph = createMatrix(V, V);
    randomPopulateGraph(graph, density);
    struct timeval creation_end;
    gettimeofday(&creation_end, NULL);
    double total_creation = ((double)creation_end.tv_sec + (double)creation_end.tv_usec / 1000000.0) - ((double)creation_start.tv_sec + (double)creation_start.tv_usec / 1000000.0);

    // Determine the grid size for CUDA kernel execution.
    computeBlockSize(V, &gridSize, &blockSize);


    // Initialize variable to measure comunication time
    struct timeval comunication_start;
    gettimeofday(&comunication_start, NULL);

    // Transfer graph data to GPU.
    CUDA_CHECK(cudaMemcpy(d_graph, graph, sizeof(TMatrix), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_items, graph->items, V * V * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&(d_graph->items), &d_items, sizeof(int*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inMST, inMST, V * sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_minEdges, minEdges, V * sizeof(TEdge), cudaMemcpyHostToDevice));

    struct timeval comunication_end;
    gettimeofday(&comunication_end, NULL);
    
    double total_comunication = ((double)comunication_end.tv_sec + (double)comunication_end.tv_usec / 1000000.0) - ((double)comunication_start.tv_sec + (double)comunication_start.tv_usec / 1000000.0); 

    // Launch the CUDA kernel for finding the MST.
    void* args[] = {
        (void*)&d_graph,
        (void*)&d_inMST,
        (void*)&d_minEdges,
        (void*)&d_MST,
        (void*)&min
    };

    // Create CUDA events for timing.
    cudaEvent_t start_gpu, end_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&end_gpu));
    CUDA_CHECK(cudaEventRecord(start_gpu));

    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)primMST, gridSize, blockSize, args));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Record the end time for GPU execution.
    CUDA_CHECK(cudaEventRecord(end_gpu));
    CUDA_CHECK(cudaEventSynchronize(end_gpu));
    float total_gpu;
    CUDA_CHECK(cudaEventElapsedTime(&total_gpu, start_gpu, end_gpu));
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(end_gpu));
    double total_prim = (double)total_gpu / 1000.0 + total_omp;

    gettimeofday(&comunication_start, NULL);
    // Transfer the MST data from GPU to CPU.
    CUDA_CHECK(cudaMemcpy(MST, d_MST, V * sizeof(TEdge), cudaMemcpyDeviceToHost));
    gettimeofday(&comunication_end, NULL);

    total_comunication += ((double)comunication_end.tv_sec + (double)comunication_end.tv_usec / 1000000.0) - ((double)comunication_start.tv_sec + (double)comunication_start.tv_usec / 1000000.0); 

    // Calculate and output timing information.
    struct timeval total_end;
    gettimeofday(&total_end, NULL);
	double total_time = ((double)total_end.tv_sec + (double)total_end.tv_usec / 1000000.0) - ((double)total_start.tv_sec + (double)total_start.tv_usec / 1000000.0);
    
    // Output the MST results to a file.
    char fileName1[200];
    sprintf(fileName1, "PrimResults/opt%d/CUDA_%d_vertices_%.2f_density_%did.txt", opt, V, density, fileNumber);
    FILE *fp1 = fopen(fileName1, "w");
    if(fp1 == NULL){
        perror("Error opening the file");
        return EXIT_FAILURE;
    }
    fprintf(fp1, "---------------------\n\nMinimum Spanning Tree\n\n");
	fprintf(fp1, "Edge \tWeight\n\n");
    for(i = 1; i < V; i ++){
        fprintf(fp1, "%d - %d \t%d \n\n", MST[i].src, MST[i].dest, MST[i].weight);
    }
    

    char fileName2[200];
    FILE *fp2;
    sprintf(fileName2, "Information/opt%d/%d_vertices_%.2f_density.csv", opt, V, density);
    fp2 = fopen(fileName2, "a+");
    if(fp2 == NULL){
        perror("Error opening the file");
        return EXIT_FAILURE;
    }

    fprintf(fp2, "CUDA+OMP;0;%d;%.07f;%.07f;%.07f;%.07f;\n", num_threads, total_creation, total_prim, total_comunication, total_time);

    // Free GPU memory and close files.
    CUDA_CHECK(cudaFree(d_graph));
    CUDA_CHECK(cudaFree(d_items));
    CUDA_CHECK(cudaFree(d_inMST));
    CUDA_CHECK(cudaFree(d_minEdges));
    CUDA_CHECK(cudaFree(d_MST));
    CUDA_CHECK(cudaFree(min));
    fclose(fp1);
    fclose(fp2);
    destroyMatrix(graph);
    free(minEdges);
    free(MST);
    free(inMST);

    return 0;
}
