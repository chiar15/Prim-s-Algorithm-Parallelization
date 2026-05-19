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
 * This header file is used to create and print a connected, undirected and weighted graph. It can be 
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

#ifndef GRAPHGENERATOR_H
#define GRAPHGENERATOR_H
#include "../Headers/TMatrix.h"

#ifdef __cplusplus
extern "C"{
#endif

/**
 * @brief Generate a random integer within the specified range [min, max].
 *
 * @param min The minimum value of the range.
 * @param max The maximum value of the range.
 * @return The random integer generated within the range.
 */
int randomInRange(int min, int max);

/**
 * @brief Populate the graph matrix with random edge weights to create a connected,
 * undirected and weighted graph.
 *
 * @param graph The matrix representing the graph (must be square)..
 * @param density The desired density of the graph (number of edges per vertex).
 */
void randomPopulateGraph(TMatrix* graph, double density);

/**
 * @brief Print the graph matrix to a file.
 *
 * @param graph The matrix representing the graph.
 * @param fp The file pointer to write the graph matrix to.
 */
void printGraph(TMatrix* graph, FILE* fp);

#ifdef __cplusplus
}
#endif
#endif