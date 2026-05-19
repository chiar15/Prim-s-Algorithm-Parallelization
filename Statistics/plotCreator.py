import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil

def calculate_speedup(data):
    # Trova il tempo di esecuzione sequenziale
    filtered_time = data[data['Modality'] == 'Sequential']['Total Time']
    sequential_time = next(iter(filtered_time), None)
    
    if sequential_time is None:
        raise ValueError("Nessun tempo sequenziale trovato. Verifica i dati.")
    
    # Calcola lo speedup
    data['SpeedUp'] = sequential_time / data['Total Time']
    return data

def plot_cuda_omp_speedup(data, output_path):
    # Crea il grafico
    omp_values = data['OMP'].unique()
    
    plt.figure(figsize=(10, 6))
    plt.xticks(np.arange(min(omp_values), max(omp_values)+1, 1.0))
    plt.yticks(np.arange(0, max(omp_values)+1, 0.5))
    plt.ylim([0, max(omp_values)+1]) 

    # Converti in NumPy array
    omp_array = np.array(omp_values)  # Converti omp_values in NumPy array
    speedup_array = data['SpeedUp'].to_numpy()  # Converti data['SpeedUp'] in NumPy array

    # Plot Speedup vs OMP Threads
    plt.plot(omp_array, speedup_array, 'b-o', label='CUDA+OMP')

    # Aggiungi la linea ideale
    plt.plot(omp_array, omp_array, 'k--', label='Ideal')

    # Imposta i titoli degli assi e la legenda
    plt.xlabel('OMP Threads')
    plt.ylabel('Speedup')
    plt.title('CUDA+OMP: Speedup vs OMP Threads')
    plt.legend()
    plt.grid(True)

    # Salva il grafico
    plt.savefig(os.path.join(output_path, 'CUDA_OMP_Speedup.png'))
    plt.close()

def plot_contour_speedup(data, output_path):
    plt.figure(figsize=(10, 6))

    # Prepara i dati per il grafico a contorno
    X, Y = np.meshgrid(data['MPI'].unique(), data['OMP'].unique())
    Z = np.array([data[(data['MPI']==x) & (data['OMP']==y)]['SpeedUp'].mean() 
                    for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    # Crea il grafico a contorno
    contour = plt.contourf(X, Y, Z, cmap='viridis')
    
    # Etichette e titoli
    plt.xlabel('MPI Processes')
    plt.ylabel('OMP Threads')
    plt.title('Contour Plot of Speedup')

    # Mostra il colorbar
    plt.colorbar(contour)

    # Salva il grafico
    plt.savefig(os.path.join(output_path, 'Contour_Speedup.png'))
    plt.close()

def generate_plots(csv_file_path, output_path):
    # Carica i dati da un file CSV
    data = pd.read_csv(csv_file_path, delimiter=';')
    data = calculate_speedup(data)

    # Filtra i dati per modalità
    mpi_openmp_data = data[data['Modality'] == 'MPI+OpenMP']
    cuda_omp_data = data[data['Modality'] == 'CUDA+OMP']

    # Genera i grafici per ogni modalità
    plot_cuda_omp_speedup(cuda_omp_data, output_path)

    # Genera grafico a contorno per MPI+OpenMP
    if not mpi_openmp_data.empty:
        plot_contour_speedup(mpi_openmp_data, output_path)

# Verifica se la directory Plots esiste, altrimenti la crea
if not os.path.exists("Plots"):
    os.mkdir("Plots")

if os.path.exists("Plots"):  # if Plots directory exist, delete it and all its elements
    shutil.rmtree('Plots')

os.mkdir("Plots")  # create the plots directory

for path, currentDirectory, files in os.walk("TimeResults"):
    for file in files:
        resultPath = path.replace("TimeResults", "Plots")
        if not(os.path.exists(resultPath)):
            os.makedirs(resultPath)
        finalPath = os.path.join(resultPath, file.replace(".csv", ""))
        if not(os.path.exists(finalPath)):
            os.makedirs(finalPath)
        generate_plots(os.path.join(path, file), finalPath)
