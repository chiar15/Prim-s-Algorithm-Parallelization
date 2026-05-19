<div align="center">
  <h1>Prim's Algorithm Parallelization</h1>

![C](https://img.shields.io/badge/C-00599C?style=for-the-badge&logo=c&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![MPI](https://img.shields.io/badge/MPI-FF6F00?style=for-the-badge&logo=openmpi&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-0298C3?style=for-the-badge&logo=openmp&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

*Parallelized Prim's algorithm with MPI, OpenMP, and CUDA. Features a performance comparison between the sequential baseline and parallelized implementations.*

[📚 Overview](#-overview) • [🗂️ Project Structure](#️-project-structure) • [⚙️ Build & Run](#️-build--run) • [📊 Performance Analysis](#-performance-analysis)
</div>

---

## 📚 Overview

This project implements Prim's algorithm for computing the **Minimum Spanning Tree (MST)** of a weighted undirected graph, developed as part of the *High Performance Computing* course at the University of Salerno. Three versions are provided and benchmarked: a **sequential** baseline written in C, a **MPI + OpenMP** hybrid version exploiting distributed and shared memory parallelism, and a **CUDA** version offloading computation to the GPU.

Each version is compiled and tested across four GCC/NVCC optimization levels (**O0** through **O3**), allowing a thorough analysis of how both parallelization and compiler optimizations affect execution time and speedup on graphs of varying sizes.

---

## 🗂️ Project Structure

```text
Prim-s-Algorithm-Parallelization/
├── Source/
│ ├── Sequential_prim.c # Sequential implementation
│ ├── MPI_OMP_prim.c # MPI + OpenMP hybrid implementation
│ ├── CUDA_prim.cu # CUDA GPU implementation
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
└── makefile # Build, test, and analysis automation
└── LICENSE
```

---

## ⚙️ Build & Run

**Prerequisites:** `gcc`, `mpicc`, `nvcc`, `python3`

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

The `make test` target runs all correctness tests and then automatically invokes the Python analysis pipeline (`analize.py`, `plotCreator.py`, `tableCreator.py`) to produce timing tables and plots.

Individual targets are also available for finer control:

| Target | Description |
|--------|-------------|
| `make compile0` – `compile3` | Compile all versions at optimization level O0–O3 |
| `make test0` – `test3` | Run correctness tests for each optimization level |
| `make analysis` | Generate plots and tables from collected timing data |

---

## 📊 Performance Analysis

Execution times were collected for all three implementations across multiple graph sizes and optimization levels. The Python scripts in `Statistics/` process the raw timing data and produce **speedup plots** and **comparison tables**, saved respectively in `Plots/` and `Tables/`.

The full methodology, experimental setup, and analysis results are documented in [HPC_Report.pdf](HPC_Report.pdf).
