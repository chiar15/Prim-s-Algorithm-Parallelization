/* 
 * Course: High Performance Computing 2023/2024
 * 
 * Lecturer: Moscato Francesco	fmoscato@unisa.it
 * 
 * Student: Ferraioli Chiara	0622702169	c.ferraioli30@studenti.unisa.it
 * 
 * 									REQUIREMENTS OF ASSIGNMENT
 * Student shall provide a parallel version of the Prim's algorithm to find the minimum spanning tree of
 * a connected, undirected and weighted matrix with both "OpenMP + MPI" and "OpenMP + Cuda" approaches, 
 * comparing results with a known solution on single-processing node. Results and differences shall be 
 * discussed for different inputs (type and size). The parallel algorithm used in "OpenMP + MPI" solution 
 * could not be the same of the "OpenMP + CUDA" approach.
 * 
 * This source file is used to create and print a connected, undirected and weighted graph. It can be 
 * used in both sequential and parallel solutions implemented with OpenMP.
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
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../Headers/graphGenerator.h"


/**
 * @brief Generate a random integer within the specified range [min, max].
 *
 * @param min The minimum value of the range.
 * @param max The maximum value of the range.
 * @return The random integer generated within the range.
 */
int randomInRange(int min, int max){
    return min + rand() % (max - min + 1);
}

/**
 * @brief Populate the adiacency matrix of a graph with random edge weights to create a connected,
 * undirected and weighted graph.
 *
 * @param graph The adiacency matrix representing the graph (must be square)..
 * @param density The desired density of the graph (number of edges per vertex).
 */
void randomPopulateGraph(TMatrix* graph, double density){
    #ifdef _OPENMP
    int thread_id = omp_get_thread_num();
    srand(time(NULL) * thread_id);
    #else
    srand(time(NULL)); 
    #endif
    int i;
    int src;
    int dest;
    int weight;
    int V = graph->cols;
    int min = 1;
    int max = 15;
    

    if(graph->cols != graph->rows){
        fprintf(stderr, "In order to be a graph the matrix has to be squared!");
        return;
    }
    
    int maxEdges = V * (V - 1) / 2;

    /*
        A weighted edge is created between each vertex and the next one (except for 
        the last vertex), in order to have a connected graph. 
        If OpenMP is enabled it uses OpenMP parallelization to create the edges, 
        allowing to use this implementation for both sequential and parallel executions.
    */
    #ifdef _OPENMP
    #pragma omp parallel for\
    default(none) shared(graph, V, min, max) private(i, weight)
    #endif
    for (i = 0; i < V - 1; ++i) {
        weight = randomInRange(min, max);
        matrixSetElement(graph, i, i+1, weight);
        matrixSetElement(graph, i+1, i, weight);
    }

    if(density < (double)2/V){
        printf("Specified density was too low! The graph has been created with the minimum number of edges needed to be connected!\n");
        return;
    }

    int numEdges = (int)(density * maxEdges);

    /*
        Edges between random vertices are created to obtain the specified density. 
        Obviously it cannot be inserted an edge between two vertices already connected.
    */

    for (i = 0; i < numEdges - (V - 1); ++i) {
        src = randomInRange(0, V - 1);

        do{
            dest = randomInRange(0, V - 1);
        } while(src == dest || matrixGetElement(graph, src, dest) != 0);

        weight = randomInRange(min, max);
        matrixSetElement(graph, src, dest, weight);
        matrixSetElement(graph, dest, src, weight);
    }
}

/**
 * @brief Print the graph matrix to a file.
 *
 * @param graph The adiacency matrix representing the graph.
 * @param fp The file pointer to write the graph matrix to.
 */
void printGraph(TMatrix* graph, FILE* fp){
    if(fp == NULL){
        return;
    }

    fprintf(fp, "Generated Graph\n\nEdge \tWeight\n\n");
    printMatrix(graph, fp);
}