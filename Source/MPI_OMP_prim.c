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
 * tree of a connected, undirected and weighted graph, realized using the "MPI+OpenMP" approach.
 * 
 * The program takes five command-line arguments: the number of omp threads, the number of vertices, 
 * graph density, optimization level and number of file.
 *
 * Usage: MPI_OMP_prim [numberOfThreads] [numberOfVertices] [graphDensity] [optimizationLevel] [fileNumber]
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

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include "../Headers/TMatrix.h"
#include"../Headers/PStructs.h"
#include "../Headers/graphGenerator.h"


/**
 * @brief Prints the Minimum Spanning Tree (MST) to a file.
 *
 * @param MST An array of edges representing the Minimum Spanning Tree.
 * @param V The number of vertices in the graph.
 * @param fp The file pointer where the MST should be printed.
 */
void printMST(TEdge* MST, int V, FILE* fp){
    if(fp == NULL){
        return;
    }
    fprintf(fp,"---------------------\n\nMinimum Spanning Tree\n\n");
	fprintf(fp, "Edge \tWeight\n\n");
    for(int i = 1; i < V; i ++){
        fprintf(fp, "%d - %d \t%d \n\n", MST[i].src, MST[i].dest, MST[i].weight);
    }
}

/**
 * @brief Computes the Minimum Spanning Tree (MST) of a connected, non-oriented and weighted graph
 * using Prim's algorithm.
 *
 * @param graph The adjacency matrix representing the graph.
 * @param MST An array to store the Minimum Spanning Tree.
 * @param d The offset for the local portion of the graph.
 * @param rank The MPI rank of the process.
 * @param inMST A boolean array representing whether a vertex is included in the MST.
 * @param comunication_total A pointer to store the total comunication time.
 */
void primMST(TMatrix* graph, TEdge* MST, int d, int rank, bool* inMST, double* comunication_total){
    int V = graph->cols;

    /*
        Number of vertices (or rows of the matrix) assigned to the process
    */
    int chunk = graph->rows;
    int i, j, k;

    /*
        An array containing the minimum edge for each vertex assigned to the process
    */
    TEdge *localEdges = (TEdge*) malloc (chunk * sizeof(TEdge)); 
    
    double comunication_start1, comunication_start2, comunication_end1, 
    comunication_end2, comunication_time1, comunication_time2;
    

    /*
        Initialize MST and inMST arrays so that only the first vertex of the graph is included
        in the Minimum Spannin Tree and initialize localEdges array
    */
    #pragma omp parallel \
    default(none) shared(V, MST, inMST, localEdges, chunk,d) private(i)
    {
        #pragma omp for
            for(i = 1; i < V; i++){
                MST[i].dest = -1;
                inMST[i] = false;
            }

        #pragma omp for
            for(i = 0; i < chunk; i++){
                localEdges[i].src = i + d;
                localEdges[i].dest = -1;
                localEdges[i].weight = INT_MAX;
            }
    }

    inMST[0] = true;


    int min; 
    int src = 0;
    int dest = 0;
    int min_key = 0;    // Last vertex inserted in the MST
    TMin minRow;    // Global minimum weight (all the mpi processes)
    TMin row;   // Local minimum weight (single mpi process)
    TEdge edge;
    int element;
    
    // Loop through V - 1 iterations to find V - 1 edges for the MST
    for (k = 0; k < V - 1; k++) {
        // Initialize the minimum weight to a high value
        min = INT_MAX;

        // Parallel loop to process vertices assigned to the current process
        #pragma omp parallel for default(none) shared(min, src, dest, min_key, inMST, localEdges, graph, chunk, d) private(i, element) 
        for (i = 0; i < chunk; i++) {

            // Check if the vertex is not already in MST
            if (!inMST[i + d]) {
                element = matrixGetElement(graph, i, min_key);

                // If an edge exists (weight not 0) and its lighter than the local edge, update localEdges
                if (element != 0 && element < localEdges[i].weight) {
                    localEdges[i].dest = min_key;
                    localEdges[i].weight = element;
                }

                // If the local edge is lighter than the minimum, update min
                if (localEdges[i].weight < min) {
                    #pragma omp critical
                    {
                        min = localEdges[i].weight;
                        src = localEdges[i].src;
                        dest = localEdges[i].dest;
                    }
                }
            }
        }

        // Create a structure to hold the minimum weight and the rank of the process
        row.min = min;
        row.rank = rank;

        // Measure and record comunication start time
        comunication_start1 = MPI_Wtime();

        // Reduce operation to find the global minimum weight and the rank of the process with the minimum weight
        MPI_Allreduce(&row, &minRow, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        // Measure and record comunication end time
        comunication_end1 = MPI_Wtime();

        // Store the source and destination of the edge with the minimum weight
        edge.src = src;
        edge.dest = dest;
        edge.weight = minRow.min;

        // Measure and record comunication start time
        comunication_start2 = MPI_Wtime();

        // Broadcast the selected edge to all processes
        MPI_Bcast(&edge, sizeof(TEdge), MPI_BYTE, minRow.rank, MPI_COMM_WORLD);

        // Measure and record comunication end time
        comunication_end2 = MPI_Wtime();

        // Update MST, inMST, and min_key based on the selected edge
        MST[edge.src].src = edge.src;
        MST[edge.src].dest = edge.dest;
        MST[edge.src].weight = edge.weight; 
        inMST[edge.src] = true;
        min_key = edge.src;

        // Calculate comunication times and update the total comunication time
        comunication_time1 = comunication_end1 - comunication_start1;
        comunication_time2 = comunication_end2 - comunication_start2;
        *comunication_total = *comunication_total + comunication_time1 + comunication_time2;

    }

    free(localEdges);
}

/**
 * @brief Main function for the parallel Prim's algorithm using MPI and OpenMP.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return The exit status.
 */
int main(int argc, char **argv) {

    int rank;
    int size;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the number of MPI processes is valid
    if (size < 0 || rank < 0) {
        fprintf(stderr, "Number of MPI processes is wrong!\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Check for the correct number of command-line arguments
    if (argc != 6) {
        fprintf(stderr, "Usage:\n\t%s [numberOfThreads] [numberOfVertices] [graphDensity] [optimizationLevel] [fileNumber]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Check if the number of threads is valid
    if (atoi(argv[1]) <= 0) {
        fprintf(stderr, "Invalid number of threads!\n");
        exit(EXIT_FAILURE);
    }

    // Check if the number of vertices is valid
    if (atoi(argv[2]) <= 0) {
        fprintf(stderr, "Invalid number of vertices!\n");
        exit(EXIT_FAILURE);
    }

    // Check if the graph density is valid (between 0 and 1)
    if (atof(argv[3]) <= 0.0 || atof(argv[3]) >= 1.0) {
        fprintf(stderr, "Invalid graph density! It must be a number between 0 and 1!\n");
        exit(EXIT_FAILURE);
    }

    // Check if the optimization level is valid (between 0 and 3)
    if (atoi(argv[4]) < 0 && atoi(argv[4]) > 3) {
        fprintf(stderr, "Invalid optimization! It must be a number between 0 and 3!\n");
        exit(EXIT_FAILURE);
    }

    if (atoi(argv[5]) < 0) {
        fprintf(stderr, "Invalid number!\n");
        exit(EXIT_FAILURE);
    }

    // Record the start time for the entire program
    double total_start = MPI_Wtime();

    int V;
    double density;
    int opt;
    int fileNumber;
    int num_threads = atoi(argv[1]);
    // Parse command-line arguments
    V = atoi(argv[2]);
    density = atof(argv[3]);
    opt = atoi(argv[4]);
    fileNumber = atoi(argv[5]);

    // Set the number of OpenMP threads
    omp_set_num_threads(num_threads);


    int *sendcounts;
    int *displs;
    int *rec_buf;
    int rem;
    int rows_per_process;
    int sum = 0;
    int rows_to_send;
    int local_V;
    int local_displs;
    double comunication_local_time;
    double comunication_total_time;
    double local_time;
    double total_time;
    double local_creation_time;
    double prim_local_time;
    double prim_total_time;
    TEdge* MST;
    TMatrix* graph;
    bool* inMST;
    char *path1 = (char*)malloc(400*sizeof(char));
    char *path2 = (char*)malloc(400*sizeof(char));
    


    // Allocate memory for MST and inMST arrays
    MST = (TEdge*) malloc(V * sizeof(TEdge));
    inMST = (bool*) malloc(V * sizeof(bool));

    /*
        Calculate sendcounts and displs arrays for scatterv operation. Every process will receive one or more
        rows (vertices) of the adiacency matrix of the graph. this calculation considers the possibility that 
        the number of vertices is not divisible by the number of threads
    */
    sendcounts = (int*) malloc(sizeof(int) * size);
    displs = (int*) malloc(sizeof(int) * size);
    rem = V % size;
    rows_per_process = V / size;
    sum = 0;

    for (int i = 0; i < size; i++) {
        rows_to_send = rows_per_process;
        if (rem > 0) {
            rows_to_send++;
            rem--;
        }

        sendcounts[i] = rows_to_send * V;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    local_V = sendcounts[rank];
    local_displs = displs[rank];

    // Allocate memory for the receive buffer
    rec_buf = (int*) malloc(local_V * sizeof(int));

    // Record the start time for graph creation
    double creation_start = MPI_Wtime();
    graph = createMatrix(V, V);
    // Create the adiacency matrix of the graph on rank 0 and populate it randomly
    if (rank == 0) {
        randomPopulateGraph(graph, density);
    }

    // Record the end time for graph creation
    double creation_end = MPI_Wtime();
    local_creation_time = creation_end - creation_start;

    // Record the start time for graph scatter
    double comunication_start = MPI_Wtime();

    // Scatter the adiacency matrix of the graph to all processes
    MPI_Scatterv(graph->items, sendcounts, displs, MPI_INT, rec_buf, local_V, MPI_INT, 0, MPI_COMM_WORLD);

    // Record the end time for graph scatter
    double comunication_end = MPI_Wtime();
    comunication_local_time = comunication_end - comunication_start;
    
    // Create a local matrix for each process to hold a part of the graph
    TMatrix* auxGraph = createMatrix(local_V / V, V);
    populateMatrix(auxGraph, rec_buf);

    // Record the start time for the Prim's algorithm
    double prim_start = MPI_Wtime();

    // Call the Prim's algorithm function
    primMST(auxGraph, MST, local_displs / V, rank, inMST, &comunication_local_time);

    // Record the end time for the Prim's algorithm
    double prim_end = MPI_Wtime();
    prim_local_time = prim_end - prim_start;

    // Record the end time for the entire program
    double total_end = MPI_Wtime();

    // Calculate local execution time
    local_time = total_end - total_start;

    // Reduce local execution times to find the maximum execution time across all processes
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&prim_local_time, &prim_total_time, 1, MPI_DOUBLE, MPI_MAX, 0,MPI_COMM_WORLD);
    MPI_Reduce(&comunication_local_time, &comunication_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_creation_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        // If rank 0, write the MST results to a file
        
        FILE *fp1;
        sprintf(path1, "PrimResults/opt%d/MPI_%d_vertices_%.2f_density_%did.txt", opt, V, density, fileNumber);
        fp1 = fopen(path1, "w");

        if (fp1 == NULL) {
            perror("Error opening the file");
            return EXIT_FAILURE;
        }
        printMST(MST, V, fp1);
        destroyMatrix(graph);
        fclose(fp1);

        // If rank 0, write timing information to a CSV file
        
        FILE* fp2;
        sprintf(path2, "Information/opt%d/%d_vertices_%.2f_density.csv", opt, V, density);
        
        char* fileName2 = path2;
        fp2 = fopen(fileName2, "a+");

        if (fp2 == NULL) {
            perror("Error opening the file");
            return EXIT_FAILURE;
        }

        fprintf(fp2, "MPI+OpenMP;");        
        fprintf(fp2, "%d;", size);
        fprintf(fp2, "%d;", num_threads);
        fprintf(fp2, "%.07f;", local_creation_time);
        fprintf(fp2, "%.07f;", prim_total_time);
        fprintf(fp2, "%.07f;", comunication_total_time);
        fprintf(fp2, "%.07f;\n", total_time);
        fflush(fp2);
        fclose(fp2);
    }

    // Clean up allocated memory and finalize MPI
    destroyMatrix(auxGraph);
    free(rec_buf);
    free(sendcounts);
    free(displs);
    free(MST);
    free(inMST);
    free(path1);
    free(path2);

    MPI_Finalize();
    return 0;
}