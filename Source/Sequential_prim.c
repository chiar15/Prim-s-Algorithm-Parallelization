/* 
 * Course: High Performance Computing 2023/2024
 * 
 * Lecturer: Moscato Francesco	fmoscato@unisa.it
 * 
 * Student: Ferraioli Chiara	0622702169	c.ferraioli30@studenti.unisa.it
 * 
 * 									REQUIREMENTS OF ASSIGNMENT
 * Student shall provide a parallel version of the Prim's algorithm to find the minimum spanning tree of
 * a connected, undirected and weighted graph with both "OpenMP + MPI" and "OpenMP + Cuda" approaches, 
 * comparing results with a known solution on single-processing node. Results and differences shall be 
 * discussed for different inputs (type and size). The parallel algorithm used in "OpenMP + MPI" solution 
 * could not be the same of the "OpenMP + CUDA" approach.
 * 
 * This source file is the sequential implementation of the Prim's algorithm to find the minimum spanning
 * tree of a connected, undirected and weighted graph. This implementation was taken from
 * https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/ and slightly adapted for 
 * the project purposes.
 * 
 * The program takes four command-line arguments: the number of vertices, graph density, optimization level
 * and number of file.
 *
 * Usage: sequential_prim [numberOfVertices] [graphDensity] [optimizationLevel] [fileNumber]
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
#include "../Headers/TMatrix.h"
#include "../Headers/graphGenerator.h"
#include <sys/time.h>


/**
 * @brief Finds the vertex with the minimum key value from the set of vertices not yet included in MST.
 *
 * @param key An array of key values for each vertex.
 * @param mstSet A boolean array representing whether a vertex is included in the MST.
 * @param V The number of vertices in the graph.
 * @return The index of the vertex with the minimum key value.
 */
int minKey(int* key, bool* mstSet, int V)
{
	// Initialize min value
	int min = INT_MAX;
    int  min_index;

	for (int v = 0; v < V; v++)
		if (mstSet[v] == false && key[v] < min)
			min = key[v], min_index = v;

	return min_index;
}

/**
 * @brief Prints the constructed MST stored in the parent[] array.
 *
 * @param parent An array representing the parent of each vertex in the MST.
 * @param graph The connected, undirected and weighted graph.
 * @param fp The file pointer to write the MST.
 */
void printMST(int* parent, TMatrix* graph, FILE* fp){

    if(fp == NULL){
        return;
    }
    fprintf(fp,"---------------------\n\nMinimum Spanning Tree\n\n");
	fprintf(fp, "Edge \tWeight\n\n");
	for (int i = 1; i < graph->cols; i++)
		fprintf(fp, "%d - %d \t%d \n\n", i, parent[i],
			matrixGetElement(graph, i, parent[i]));
}

/**
 * @brief Constructs and prints the Minimum Spanning Tree (MST) for a connected, undirected and weighted graph.
 *
 * @param graph The connected, undirected and weighted graph represented as a TMatrix.
 * @param fp The file pointer to write the MST.
 */
void primMST(TMatrix* graph, int *parent, int *key, bool* mstSet){

    int V = graph->cols;
	

	// Initialize all keys as INFINITE
	for (int i = 0; i < V; i++)
		key[i] = INT_MAX, mstSet[i] = false;

	/*
		Always include first 1st vertex in MST.
		Make key 0 so that this vertex is picked as first
		vertex.
	*/ 
	key[0] = 0;

	// First node is always root of MST
	parent[0] = -1;

	// The MST will have V vertices
	for (int count = 0; count < V - 1; count++) {
		
		// Pick the minimum key vertex from the
		// set of vertices not yet included in MST
		int u = minKey(key, mstSet, V);

		// Add the picked vertex to the MST Set
		mstSet[u] = true;

		/* 
			Update key value and parent index of
			the adjacent vertices of the picked vertex.
			Consider only those vertices which are not
			yet included in MST
		*/
		for (int v = 0; v < V; v++)

			/*
				graph[u][v] is non zero only for adjacent
				vertices of m mstSet[v] is false for vertices
				not yet included in MST Update the key only
				if graph[u][v] is smaller than key[v]
			*/
			if (matrixGetElement(graph, u, v) && mstSet[v] == false
				&& matrixGetElement(graph, u, v) < key[v])
				parent[v] = u, key[v] = matrixGetElement(graph, u, v);
	}

}

/**
 * @brief The main function that parses command-line arguments, generates a connected, undirected and weighted
 * graph, and runs the Prim's algorithm to find the MST.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return An integer representing the exit status.
 */
int main(int argc,char **argv){

	if(argc != 5){
        fprintf(stderr, "Usage:\n\t%s [numberOfVertices] [graphDensity] [optimizationLevel] [fileNumber]\n", argv[0]);
		exit(EXIT_FAILURE);
    }

	if(atoi(argv[1]) <= 0){
		fprintf(stderr, "Invalid number of vertices!\n");
		exit(EXIT_FAILURE);
	}

	if(atof(argv[2]) <= 0.0 || atof(argv[2]) >= 1.0){
		printf("%f", atof(argv[2]));
		fprintf(stderr, "Invalid graph density! It must be a number between 0 and 1!\n");
		exit(EXIT_FAILURE);
	}

	if(atoi(argv[3]) < 0 && atoi(argv[3]) < 3){
		fprintf(stderr, "Invalid optimization! It must be a number between 0 and 3!\n");
		exit(EXIT_FAILURE);
	}

	if(atoi(argv[4]) < 0){
		fprintf(stderr, "Invalid number!\n");
		exit(EXIT_FAILURE);
	}

	/*
		Get the start time to calculate the running time of the entire program
	*/
	struct timeval total_start, total_end;
	double total_time;
    gettimeofday(&total_start, NULL);
	
	// Parse command-line arguments
	int V = atoi(argv[1]);

	double density = atof(argv[2]);
	
	int opt = atoi(argv[3]);

	int fileNumber = atoi(argv[4]);

	// Array to store constructed MST
	int* parent = (int*) malloc(V * sizeof(int));
	// Key values used to pick minimum weight edge in cut
	int* key = (int*) malloc(V * sizeof(int));
	// To represent set of vertices included in MST
	bool* mstSet = (bool*) malloc(V * sizeof(bool));

	/*
		Get the start time to calculate the running time of the creation of the graph
	*/
	struct timeval creation_start, creation_end;
	double creation_time;
    gettimeofday(&creation_start, NULL);

	/*
		Creation of a square matrix to be used as graph and its population
	*/
	TMatrix* graph = createMatrix(V, V);
	randomPopulateGraph(graph, density);

	/*
		Get the end time to calculate the running time of the creation of the graph
	*/
	gettimeofday(&creation_end, NULL);

	/*
		Calculate the running time of the creation of the graph in term of seconds
	*/
	creation_time = ((double)creation_end.tv_sec + (double)creation_end.tv_usec / 1000000.0) - ((double)creation_start.tv_sec + (double)creation_start.tv_usec / 1000000.0);

	/*
		Get the start time to calculate the running time of the Prim's algorithm
	*/
	struct timeval prim_start, prim_end;
	double prim_time;
	gettimeofday(&prim_start, NULL);

	// Call the Prim's algorithm function
	primMST(graph, parent, key, mstSet);

	/*
		Get the end time to calculate the running time of the Prim's algorithm
	*/
	gettimeofday(&prim_end, NULL);

	/*
		Calculate the running time of the Prim's algorithm in term of seconds
	*/
	prim_time = ((double)prim_end.tv_sec + (double)prim_end.tv_usec / 1000000.0) - ((double)prim_start.tv_sec + (double)prim_start.tv_usec / 1000000.0);
    
	/*
		Get the end time to calculate the running time of the entire program
	*/
	gettimeofday(&total_end, NULL);

	/*
		Calculate the running time of the entire program in term of seconds
	*/
	total_time = ((double)total_end.tv_sec + (double)total_end.tv_usec / 1000000.0) - ((double)total_start.tv_sec + (double)total_start.tv_usec / 1000000.0);

	char filePath1[200];
	sprintf(filePath1, "PrimResults/opt%d/Seq_%d_vertices_%.2f_density_%did.txt", opt, V, density, fileNumber);
    FILE *fp1 = fopen(filePath1, "w");

    if(fp1 == NULL){
        perror("Error opening the file");
        return EXIT_FAILURE;
    }

	// print the constructed MST
	printMST(parent, graph, fp1);
	/*
		Save the time results on a .csv file
	*/
	char fileName2[200];
    FILE *fp2;

	sprintf(fileName2, "Information/opt%d/%d_vertices_%.2f_density.csv", opt, V, density);
	fp2 = fopen(fileName2, "a+");

    if(fp2 == NULL){
        perror("Error opening the file");
        return EXIT_FAILURE;
    }

	
	fprintf(fp2, "Sequential;0;0; %.07f; %.07f; 0.0; %.07f;\n", creation_time , prim_time, total_time);
	free(parent);
	free(key);
	free(mstSet);
	fclose(fp1);
	fclose(fp2);
	return 0;
}
