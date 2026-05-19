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
 * This header file is used to define some structs useful for the parallel implementations of the Prim's
 * algorithm.
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

#ifndef PSTRUCTS_H
#define PSTRUCTS_H

#ifdef __cplusplus
extern "C"{
#endif

/**
 * @brief Structure to represent an edge with source, destination, and weight.
 */
typedef struct SEdge{
    int src;    // Source of the edge
    int dest;   // Destination of the edge
    int weight; // Weight of the edge
} TEdge;

/**
 * @brief Structure to represent the minimum value along with its associated rank used in the parallel algorithm.
 */
typedef struct SMin{
    int min;    // Minimum value
    int rank;   // Rank associated with the minimum value
} TMin;

#ifdef __cplusplus
}
#endif

#endif