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
 * This header file is used to manage dynamic matrixes. It can be used in both sequential and parallel 
 * solutions implemented with OpenMP.
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

#ifndef TMATRIX_H
#define TMATRIX_H
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"{
#endif
/**
 * @brief Structure to represent a dinamic matrix
 */
struct SMatrix {
    int *items; // Pointer to the matrix elements.
    int rows;   // The number of rows in the matrix.
    int cols;   // The number of columns in the matrix.
};

typedef struct SMatrix TMatrix; // Alias for the matrix structure.

/**
 * @brief Create a new dynamic matrix with the specified number of rows and columns.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @return A pointer to the newly created matrix.
 */
TMatrix* createMatrix(int rows, int cols);

/**
 * @brief Get the value of an element in the matrix at the specified row and column.
 *
 * @param matrix The matrix.
 * @param row The row index.
 * @param col The column index.
 * @return The value at the specified row and column.
 */
int matrixGetElement(TMatrix* matrix, int row, int col);

/**
 * @brief Set the value of an element in the matrix at the specified row and column.
 *
 * @param matrix The matrix.
 * @param row The row index.
 * @param col The column index.
 * @param element The value to set.
 */
void matrixSetElement(TMatrix* matrix, int row, int col, int element);

/**
 * @brief Populate the matrix with values from an auxiliary matrix.
 *
 * @param matrix The matrix to populate.
 * @param aux_matrix The auxiliary matrix containing the values.
 */
void populateMatrix(TMatrix* matrix, int* aux_matrix);

/**
 * @brief Print the matrix to a file.
 *
 * @param matrix The matrix to print.
 * @param fp The file pointer to write the matrix to.
 */
void printMatrix(TMatrix* matrix, FILE* fp);

/**
 * @brief Deallocate the memory used by a matrix.
 *
 * @param matrix The matrix to destroy.
 */
void destroyMatrix(TMatrix* matrix);

#ifdef __cplusplus
}
#endif
#endif