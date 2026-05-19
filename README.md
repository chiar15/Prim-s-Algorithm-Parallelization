<div align="center">
  <h1>Prim's Algorithm Parallelization</h1>

![C](https://img.shields.io/badge/C-00599C?style=for-the-badge&logo=c&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![MPI](https://img.shields.io/badge/MPI-FF6F00?style=for-the-badge&logo=openmpi&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-0298C3?style=for-the-badge&logo=openmp&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

*Parallelized Prim's algorithm with MPI, OpenMP, and CUDA. Features a performance comparison between the sequential baseline and parallelized implementations.*

[📚 Overview](#-overview) • [📖 Theoretical Background](#-theoretical-background) • [⚙️ Implementation](#️-implementation) • [🖥️ Experimental Setup](#️-experimental-setup) • [🗂️ Project Structure](#️-project-structure) • [🔧 Prerequisites](#-prerequisites) • [🛠️ Build & Run](#️-build--run) • [📊 Performance Analysis](#-performance-analysis)
</div>

---

## 📚 Overview

This project investigates the parallelization of **Prim's algorithm** for computing the **Minimum Spanning Tree (MST)** of a connected, undirected, and weighted graph. Three implementations are provided and benchmarked: a **sequential** baseline written in C, a **MPI + OpenMP** hybrid version exploiting distributed and shared memory parallelism, and a **CUDA + OpenMP** version offloading computation to the GPU.

Each version is compiled and tested across four GCC/NVCC optimization levels (**O0** through **O3**), allowing a thorough analysis of how both parallelization strategies and compiler optimizations affect execution time and speedup on graphs of varying sizes and densities.

---

## 📖 Theoretical Background

**Prim's algorithm** is a greedy algorithm that builds the MST by iteratively selecting the minimum weight edge connecting a vertex already in the tree to one outside it. Its sequential complexity is \(O(V^2)\) for dense graphs represented as adjacency matrices, where \(V\) is the number of vertices.

Parallelization is explored through three distinct paradigms:
- **OpenMP** enables shared-memory multi-threading, distributing independent loop iterations across CPU cores with minimal synchronization overhead.
- **MPI** introduces distributed parallelism through message passing, partitioning the graph across processes and using collective operations (e.g. `MPI_Allreduce`) to coordinate the global minimum selection at each step.
- **CUDA** leverages thousands of lightweight GPU cores to parallelize the most compute-intensive phases, such as the search for the minimum key vertex, offsetting the cost of host-device memory transfers on large graphs.

Note that the parallel algorithm used in the MPI + OpenMP solution is intentionally different from the one adopted in the CUDA + OpenMP approach, as required by the assignment.

---

## ⚙️ Implementation

The project is organized into dedicated source files, each separating a specific concern:

`Sequential_prim.c` — Baseline sequential implementation of Prim's algorithm. Serves as the reference for speedup calculations. Takes four command-line arguments: number of vertices, graph density, optimization level, and file number.

`MPI_OMP_prim.c` — Hybrid parallel implementation combining MPI for inter-process distribution and OpenMP for intra-process threading. The graph rows are distributed among MPI processes; at each iteration, each process finds the local minimum, and a global reduction selects the overall minimum vertex.

`CUDA_prim.cu` — GPU-accelerated implementation using CUDA kernels for the minimum key search and key update phases, combined with OpenMP on the host side. Includes optimizations such as shared memory usage within kernels.

`TMatrix.c` — Utility functions for the adjacency matrix data structure (`TMatrix`), used as the graph representation across all implementations.

`graphGenerator.c` — Random weighted graph generator that populates a `TMatrix` with a given number of vertices and edge density.

`Statistics/analize.py`, `plotCreator.py`, `tableCreator.py` — Python post-processing scripts that read raw CSV timing data and produce speedup plots and comparison tables.

---

## 🖥️ Experimental Setup

### 🛠️ Software
- **Compilers:** GCC, MPICC, NVCC (CUDA Toolkit)
- **Parallelism:** OpenMP, MPI, CUDA
- **Analysis:** Python 3 with `matplotlib`, `pandas`, `numpy`

---

## 🗂️ Project Structure

```text
Prim-s-Algorithm-Parallelization/
├── Source/
│ ├── Sequential_prim.c # Sequential implementation
│ ├── MPI_OMP_prim.c # MPI + OpenMP hybrid implementation
│ ├── CUDA_prim.cu # CUDA + OpenMP GPU implementation
│ ├── TMatrix.c # Adjacency matrix data structure
│ ├── graphGenerator.c # Random weighted graph generator
│ └── directories.py # Build directory setup script
├── Headers/ # Header files
├── Test/ # Correctness test scripts (test0–test3)
├── Statistics/ # Python scripts for analysis and plots
│ ├── analize.py
│ ├── plotCreator.py
│ └── tableCreator.py
├── HPC_Report.pdf # Full project report
├── LICENSE
└── makefile # Build, test, and analysis automation
```

---

## 🔧 Prerequisites

Before running the Makefile commands, ensure you have the following installed:

- **GCC** — for C code compilation
- **MPICC** — for MPI parallel execution
- **NVCC** — from the CUDA Toolkit, for GPU-based computation
- **OpenMP** — for multi-threading support (included in GCC)
- **Python 3** with:
  - `matplotlib` — for plots
  - `pandas` — for data processing
  - `numpy` — for numerical computations

---

## 🛠️ Build & Run

**Build all versions** (O0 through O3):
```bash
make all
```

**Run all tests and generate statistics:**
```bash
make test
```

**Clean all build artifacts:**
```bash
make clean
```

The `make test` target runs all correctness tests and then automatically invokes the Python analysis pipeline to produce timing tables and plots.

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make compile0` – `compile3` | Compile all versions at optimization level O0–O3 |
| `make test0` – `test3` | Run correctness tests for each optimization level |
| `make analysis` | Generate plots and tables from collected timing data |

---

## 📊 Performance Analysis

Execution times were collected for all three implementations across multiple graph sizes, densities, and optimization levels. The Python scripts in `Statistics/` process the raw CSV timing data and produce **speedup plots** and **comparison tables**, saved respectively in `Plots/` and `Tables/`.

Key metrics evaluated:
- **Execution time** for graph creation and MST computation separately
- **Speedup** of MPI+OpenMP and CUDA+OpenMP versions relative to the sequential baseline
- **Scalability** across increasing graph sizes and varying densities

The full methodology, experimental setup, and analysis results are documented in [HPC_Report.pdf](HPC_Report.pdf).
