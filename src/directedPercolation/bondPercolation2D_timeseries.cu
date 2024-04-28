// CUDA C++
#include <cuda.h>
#include <curand_kernel.h>
#include <random>
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

// Define constants
constexpr double P = 0.3845;
constexpr int L = 1024;
constexpr int N_STEPS = 4000;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, 1);

std::vector<bool> initLattice(int L)
{
//     std::vector<bool> soil_lattice(L * L);
//     for (int i = 0; i < L * L; ++i)
//     {
//         soil_lattice[i] = dis(gen);
//     }
//     return soil_lattice;

    std::vector<bool> soil_lattice(L * L, false); // All sites off
    soil_lattice[L * L / 2 + L / 2] = true;       // Middle site on
    return soil_lattice;
}

__global__ void initCurand(curandState *state, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Initialize the RNG state for this thread
    curand_init(seed, index, 0, &state[index]);
}

__global__ void updateKernel(bool *d_lattice, bool *d_latticeUpdated, curandState *state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Initialize the RNG
    curandState localState = state[index]; // Copy the state to local memory for efficiency

    int nPercolationTrials = 0;

    for (int i = 0; i < 4; ++i)
    {
        int x = idx + (i % 2) * 2 - 1;
        int y = idy + (i / 2) * 2 - 1;

        // Periodic boundary conditions
        x = (x + L) % L;
        y = (y + L) % L;

        // Check if the site is occupied
        if (d_lattice[x + y * L])
        {
            nPercolationTrials++;
        }
    }

    if (nPercolationTrials > 0)
    {
        if (curand_uniform(&localState) < 1 - pow(1 - P, nPercolationTrials))
        {
            d_latticeUpdated[idx + idy * L] = true;
        }
    }

    // Update the state
    state[index] = localState;
}

__global__ void countLattice(bool *d_lattice, int *d_counts, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size && d_lattice[idx])
    {
        atomicAdd(d_counts, 1);
    }
}

__global__ void normalizeCounts(int *d_countArray, float *d_normalizedArray)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sites = L * L;

    if (idx < N_STEPS * 5)
    {
        d_normalizedArray[idx] = static_cast<float>(d_countArray[idx]) / total_sites;
    }
}

void run(std::ofstream &file)
{
    // Initialize the lattice
    std::vector<bool> soil_lattice = initLattice(L);

    // Initialize CUDA
    cudaSetDevice(0);

    // Allocate memory on the GPU
    bool *d_lattice;
    bool *d_latticeUpdated;
    curandState *d_state;
    cudaMalloc(&d_lattice, L * L * sizeof(bool));
    cudaMalloc(&d_latticeUpdated, L * L * sizeof(bool));
    cudaMalloc(&d_state, L * L * sizeof(curandState));

    // Initialize the RNG states
    initCurand<<<L, L>>>(d_state, time(0));

    // Copy the lattice data to a temporary std::vector<char> for the cudaMemcpy call
    std::vector<char> temp_lattice(soil_lattice.begin(), soil_lattice.end());

    // Copy the lattice data to the GPU
    cudaMemcpy(d_lattice, temp_lattice.data(), L * L * sizeof(char), cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    dim3 blockSize(1, 1);
    dim3 gridSize(L, L);

    // Initialize counts to 0
    int *d_countArray;
    cudaMalloc(&d_countArray, N_STEPS * sizeof(int));
    cudaMemset(d_countArray, 0, N_STEPS * sizeof(int));

    int threadsPerBlockCounting = 256;
    int blocksPerGridCounting = (L * L + threadsPerBlockCounting - 1) / threadsPerBlockCounting;

    // Launch the CUDA kernel for each of the A, B, C, and D squares
    for (int step = 0; step < N_STEPS; ++step)
    {
        // reset the updated lattice to all zeros
        cudaMemset(d_latticeUpdated, 0, L * L * sizeof(bool));

        updateKernel<<<gridSize, blockSize>>>(d_lattice, d_latticeUpdated, d_state);
        cudaDeviceSynchronize();

        cudaMemcpy(d_lattice, d_latticeUpdated, L * L * sizeof(bool), cudaMemcpyDeviceToDevice);

        countLattice<<<blocksPerGridCounting, threadsPerBlockCounting>>>(d_latticeUpdated, d_countArray + step, L * L);

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / (N_STEPS - 1) * 100 << "%\r" << std::flush;
    }

    float *d_normalizedArray;
    cudaMalloc(&d_normalizedArray, N_STEPS * sizeof(float));

    normalizeCounts<<<blocksPerGridCounting, threadsPerBlockCounting>>>(d_countArray, d_normalizedArray);
    cudaDeviceSynchronize();

    std::vector<float> counts(N_STEPS);
    cudaMemcpy(counts.data(), d_normalizedArray, N_STEPS * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_STEPS; ++i)
    {
        file << counts[i] << "\n";
    }

    // Free the memory on the GPU
    cudaFree(d_lattice);
    cudaFree(d_latticeUpdated);
    cudaFree(d_state);
    cudaFree(d_countArray);
    cudaFree(d_normalizedArray);
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    // filePathStream << exeDir << "/outputs/timeseries2D/randomIC/p_" << P << ".csv";
    filePathStream << exeDir << "/outputs/timeseries2D/seedIC/p_" << P << ".csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    run(file);
    file.close();

    return 0;
}