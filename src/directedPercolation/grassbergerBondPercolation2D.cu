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
constexpr double P = 0.2873;
constexpr int L = 1024;
constexpr int N_STEPS = 1000;
constexpr int N_SIMULATIONS = 10000;

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

__global__ void countLattice(bool *d_lattice, int *d_counts, int *d_activeSites)
{
    extern __shared__ int local_counts[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    local_counts[threadIdx.x] = (idx < L*L && d_lattice[idx]) ? 1 : 0;

    __syncthreads();

    // Perform reduction in each block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            local_counts[threadIdx.x] += local_counts[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (threadIdx.x == 0)
    {
        atomicAdd(d_counts, local_counts[0]);
        atomicAdd(d_activeSites, local_counts[0]);
    }
}

__global__ void checkZero(int *count, bool *flag)
{
    if (*count == 0)
    {
        *flag = true;
    }
}

__global__ void calculateR2(bool *d_lattice, double *d_preCalcDistances, int *d_activeSites, double *d_R2)
{
    extern __shared__ double sdata[];

    // Each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < L*L && d_lattice[i]) ? d_preCalcDistances[i] : 0;

    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        d_R2[blockIdx.x] = sdata[0];
    }

    // Calculate mean square distance
    if (blockIdx.x == 0 && tid == 0 && *d_activeSites > 0)
    {
        double total = 0.0;
        for (int i = 0; i < gridDim.x; i++)
        {
            total += d_R2[i];
        }
        *d_R2 = total / *d_activeSites;
    }
}

void run(std::ofstream &file)
{
    // Initialize CUDA
    cudaSetDevice(0);

    // Allocate memory on the GPU
    bool *d_lattice;
    bool *d_latticeUpdated;
    curandState *d_state;
    int *d_countArray;
    double *d_squareDistances;
    int *d_activeSites;
    double *d_R2;
    bool *h_simulationDeadFlag;

    cudaMalloc(&d_lattice, L * L * sizeof(bool));
    cudaMalloc(&d_latticeUpdated, L * L * sizeof(bool));
    cudaMalloc(&d_state, L * L * sizeof(curandState));
    cudaMalloc(&d_countArray, (N_STEPS + 1) * sizeof(int));
    cudaMalloc(&d_squareDistances, L * L * sizeof(double));
    cudaMalloc(&d_activeSites, sizeof(int));
    cudaMalloc(&d_R2, (N_STEPS + 1) * sizeof(double));
    cudaHostAlloc((void **)&h_simulationDeadFlag, sizeof(bool), cudaHostAllocDefault);

    std::vector<double> squareDistances(L * L);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            int x = i - L / 2;
            int y = j - L / 2;
            squareDistances[i * L + j] = static_cast<double>(x * x + y * y);
        }
    }

    cudaMemcpy(d_squareDistances, squareDistances.data(), L * L * sizeof(double), cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    dim3 blockSize(1, 1);
    dim3 gridSize(L, L);

    int threadsPerBlockCounting = 256;
    int blocksPerGridCounting = (L * L + threadsPerBlockCounting - 1) / threadsPerBlockCounting;

    for (int simulationNumber = 0; simulationNumber < N_SIMULATIONS; ++simulationNumber)
    {
        // Initialize the lattice
        std::vector<bool> soil_lattice(L * L, false); // All sites off
        soil_lattice[L * L / 2 + L / 2] = true;       // Middle site on
        soil_lattice[L * (L / 2 + 1) + L / 2] = true;
        soil_lattice[L * L / 2 + (L / 2 + 1)] = true;
        soil_lattice[L * (L / 2 + 1) + (L / 2 + 1)] = true;

        // Initialize the RNG states
        initCurand<<<L, L>>>(d_state, time(0) + simulationNumber);

        // Copy the lattice data to a temporary std::vector<char> for the cudaMemcpy call
        std::vector<char> temp_lattice(soil_lattice.begin(), soil_lattice.end());

        // Copy the lattice data to the GPU
        cudaMemcpy(d_lattice, temp_lattice.data(), L * L * sizeof(char), cudaMemcpyHostToDevice);

        // Initialize counts to 0
        cudaMemset(d_countArray, 0, N_STEPS * sizeof(int));
        cudaMemset(d_activeSites, 0, sizeof(int));
        cudaMemset(d_R2, 0, N_STEPS * sizeof(double));
        *h_simulationDeadFlag = false;

        countLattice<<<blocksPerGridCounting, threadsPerBlockCounting, threadsPerBlockCounting * sizeof(int)>>>(d_lattice, d_countArray, d_activeSites);
        cudaDeviceSynchronize();

        calculateR2<<<blocksPerGridCounting, threadsPerBlockCounting, threadsPerBlockCounting * sizeof(double)>>>(d_lattice, d_squareDistances, d_activeSites, d_R2);
        cudaDeviceSynchronize();

        for (int step = 1; step < N_STEPS; ++step)
        {
            // reset the updated lattice to all zeros
            cudaMemset(d_latticeUpdated, 0, L * L * sizeof(bool));

            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_latticeUpdated, d_state);
            cudaDeviceSynchronize();

            cudaMemcpy(d_lattice, d_latticeUpdated, L * L * sizeof(bool), cudaMemcpyDeviceToDevice);

            cudaMemset(d_activeSites, 0, sizeof(int));
            countLattice<<<blocksPerGridCounting, threadsPerBlockCounting, threadsPerBlockCounting * sizeof(int)>>>(d_latticeUpdated, d_countArray + step, d_activeSites);
            cudaDeviceSynchronize();

            checkZero<<<1, 1>>>(d_countArray + step, h_simulationDeadFlag);
            cudaDeviceSynchronize();

            if (*h_simulationDeadFlag)
            {
                break;
            }

            calculateR2<<<blocksPerGridCounting, threadsPerBlockCounting, threadsPerBlockCounting * sizeof(double)>>>(d_lattice, d_squareDistances, d_activeSites, d_R2 + step);
            cudaDeviceSynchronize();

        }

        std::vector<int> counts(N_STEPS + 1);
        cudaMemcpy(counts.data(), d_countArray, (N_STEPS + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<double> R2(N_STEPS + 1);
        cudaMemcpy(R2.data(), d_R2, (N_STEPS + 1) * sizeof(double), cudaMemcpyDeviceToHost);
        // write the simulation number, time, and counts to the file
        for (int i = 0; i < N_STEPS + 1; ++i)
        {
            if (counts[i] == 0)
            {
                break;
            }
            file << simulationNumber << "," << i << "," << counts[i] << "," << R2[i] << "\n";
        }

        std::cout << "Progress: " << static_cast<double>(simulationNumber + 1) / N_SIMULATIONS * 100 << "%\r" << std::flush;
    }

    // Free the memory on the GPU
    cudaFree(d_lattice);
    cudaFree(d_latticeUpdated);
    cudaFree(d_state);
    cudaFree(d_countArray);
    cudaFreeHost(h_simulationDeadFlag);
    cudaFree(d_R2);
    cudaFree(d_squareDistances);
    cudaFree(d_activeSites);
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/grassberger/4IC_p_" << P << "_L_" << L << ".csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "simulation,time,activeCounts,R2\n";

    run(file);

    file.close();

    return 0;
}