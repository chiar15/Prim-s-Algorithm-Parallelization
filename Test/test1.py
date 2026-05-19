import subprocess
import os
import time

def run_command(command):
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        print("Error in command execution:", command)
        try:
            print(process.stderr.decode('utf-8', errors='replace'))
        except UnicodeDecodeError:
            print("Non-UTF-8 error output, not displayed.")
    else:
        try:
            print(process.stdout.decode('utf-8', errors='replace'))
        except UnicodeDecodeError:
            print("Non-UTF-8 output, not displayed.")
    

# Ottenere il percorso assoluto della directory di lavoro (cartella "Test")
current_directory = os.path.abspath(os.getcwd())

vertexes = {500, 2000, 10000}
density = {0.25, 0.50, 0.75}
mpi_max = 4
omp_max = 8
iterations = 5
opt = 1

start_time = time.time()

for v in vertexes:
    for d in density:
        for i in range(iterations):
            # Utilizzare il percorso relativo per il primo e il secondo comando
            command_seq = os.path.join(current_directory, 'Build', 'Sequential_prim1.exe')
            command = f'{command_seq} {v} {d} {opt} {i}'
            print(command)
            run_command(command)
            omp = 1
            while omp <= omp_max:
                # Utilizzare il percorso relativo per il terzo comando
                command_cuda = os.path.join(current_directory, 'Build', 'CUDA_prim1.exe')
                command = f'{command_cuda} {omp} {v} {d} {opt} {i}'
                print(command)
                run_command(command)
                mpi = 1
                while mpi <= mpi_max and (omp * mpi) < 12:
                    # Utilizzare il percorso relativo per il quarto comando
                    command_mpi = os.path.join(current_directory, 'Build', 'MPI_OMP_prim1.exe')
                    command = f'mpirun -np {mpi} {command_mpi} {omp} {v} {d} {opt} {i}'
                    print(command)
                    run_command(command)
                    mpi = mpi * 2
                omp = omp * 2

total_time_seconds = time.time() - start_time

total_time_minutes = total_time_seconds / 60


print(f"Test 1: {total_time_minutes:.2f} minuti")