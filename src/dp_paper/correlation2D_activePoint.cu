// CUDA C++
#include <cuda.h>
#include <curand_kernel.h>
#include <chrono>
#include <random>
#include <vector>
#include <array>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <float.h>

// Define constants
constexpr double P = 0.318;
constexpr int L = 1024;
constexpr int N_STEPS = 2000;
constexpr int RECORDING_STEP = N_STEPS * 5 / 10;
constexpr int ZERO = 0; // yes. zero. for real. send help.
constexpr int threadsPerBlockCounting = 256;
constexpr int blocksPerGridCounting = (L * L + threadsPerBlockCounting - 1) / threadsPerBlockCounting;

std::vector<bool> initLattice(int L)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    std::vector<bool> soil_lattice(L * L, false); // Initialize all cells to false
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            soil_lattice[i * L + j] = dis(gen);
        }
    }
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

__global__ void updateKernel(bool *d_lattice, bool *d_latticeUpdated, double p, curandState *state)
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
        int x = idx + (i % 2);
        int y = idy + (i / 2);

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
        if (curand_uniform(&localState) < 1 - pow(1 - p, nPercolationTrials))
        {
            d_latticeUpdated[idx + idy * L] = true;
        }
    }

    // Update the state
    state[index] = localState;
}

__global__ void calculateDistances(bool *lattice, int *distances, int *activePoint)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


    if (i < L && j < L)
    {
        if (lattice[i * L + j])
        {
            int dx = abs(*activePoint / L - i);
            int dy = abs(*activePoint % L - j);
            dx = min(dx, L - dx); // Account for periodic boundary conditions
            dy = min(dy, L - dy); // Account for periodic boundary conditions
            distances[i * L + j] = dx + dy;
        }
        else
        {
            distances[i * L + j] = -1;
        }
    }
}

__global__ void calculateHistogram(int *distances, float *histogram)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x;
    if (i < L * L)
    {
        int distance = distances[i];
        if (distance != -1)
        {
            atomicAdd(&histogram[(int)distance], 1.0f);
        }
    }
}

__global__ void countLattice(bool *d_lattice, int *d_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < L * L && d_lattice[idx])
    {
        atomicAdd(d_counts, 1);
    }
}

__global__ void normalizeHistogram(float *histogram, int *num_active_sites)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    histogram[i] /= ((float)(*num_active_sites - 1) / (float)(L * L));
}

__global__ void addToTotalHistogram(float *histogram, float *histogram_total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    histogram_total[i] += histogram[i];
}

__global__ void generateRandomNumbers(bool *lattice, float *randomNumbers, curandState *states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < L && j < L)
    {
        int index = i * L + j;
        if (lattice[index])
        {
            randomNumbers[index] = curand_uniform(&states[index]);
        }
        else
        {
            randomNumbers[index] = -1.0f;
        }
    }
}

// Global variables for block-level reduction
__device__ float blk_vals[blocksPerGridCounting]; // Adjust size as needed
__device__ int blk_idxs[blocksPerGridCounting];   // Adjust size as needed
__device__ int blk_num = 0;

__global__ void findMaxIndex(const float *data, int *result)
{
    extern __shared__ char shared_memory[];                 // Use dynamic shared memory
    volatile float *vals = (volatile float *)shared_memory; // Cast part of shared memory to float array for vals
    volatile int *idxs = (volatile int *)&vals[blockDim.x]; // Cast another part for idxs, right after vals
    __shared__ volatile int last_block;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    last_block = 0;
    float my_val = -FLT_MAX;
    int my_idx = -1;
    while (idx < L * L)
    {
        if (data[idx] > my_val)
        {
            my_val = data[idx];
            my_idx = idx;
        }
        idx += blockDim.x * gridDim.x;
    }
    vals[threadIdx.x] = my_val;
    idxs[threadIdx.x] = my_idx;
    __syncthreads();
    for (int i = (blockDim.x >> 1); i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
            if (vals[threadIdx.x] < vals[threadIdx.x + i])
            {
                vals[threadIdx.x] = vals[threadIdx.x + i];
                idxs[threadIdx.x] = idxs[threadIdx.x + i];
            }
        __syncthreads();
    }
    if (!threadIdx.x)
    {
        blk_vals[blockIdx.x] = vals[0];
        blk_idxs[blockIdx.x] = idxs[0];
        if (atomicAdd(&blk_num, 1) == gridDim.x - 1)
            last_block = 1;
    }
    __syncthreads();
    if (last_block)
    {
        idx = threadIdx.x;
        my_val = -FLT_MAX;
        my_idx = -1;
        while (idx < gridDim.x)
        {
            if (blk_vals[idx] > my_val)
            {
                my_val = blk_vals[idx];
                my_idx = blk_idxs[idx];
            }
            idx += blockDim.x;
        }
        vals[threadIdx.x] = my_val;
        idxs[threadIdx.x] = my_idx;
        __syncthreads();
        for (int i = (blockDim.x >> 1); i > 0; i >>= 1)
        {
            if (threadIdx.x < i)
                if (vals[threadIdx.x] < vals[threadIdx.x + i])
                {
                    vals[threadIdx.x] = vals[threadIdx.x + i];
                    idxs[threadIdx.x] = idxs[threadIdx.x + i];
                }
            __syncthreads();
        }
        if (!threadIdx.x)
            *result = idxs[0];
    }
}

void run(std::ofstream &file, double p)
{
    std::vector<bool> soil_lattice = initLattice(L);

    cudaSetDevice(0);

    bool *d_lattice;
    bool *d_latticeUpdated;
    curandState *d_state;
    int *d_distances;
    float *d_histogram;
    float *d_histogram_total;
    int *d_nActiveSites;
    float *d_randomNumbers;
    int *d_maxIndex;
    cudaMalloc(&d_lattice, L * L * sizeof(bool));
    cudaMalloc(&d_latticeUpdated, L * L * sizeof(bool));
    cudaMalloc(&d_state, L * L * sizeof(curandState));
    cudaMalloc(&d_distances, L * L * sizeof(int));
    cudaMalloc(&d_histogram, L * sizeof(float));
    cudaMalloc(&d_histogram_total, L * sizeof(float));
    cudaMalloc(&d_nActiveSites, sizeof(int));
    cudaMalloc(&d_randomNumbers, L * L * sizeof(float));
    cudaMalloc(&d_maxIndex, sizeof(int));

    cudaMemset(d_histogram_total, 0, L * sizeof(float));

    dim3 blockSize(1, 1);
    dim3 gridSize(L, L);

    initCurand<<<gridSize, blockSize>>>(d_state, time(0));

    std::vector<char> temp_lattice(soil_lattice.begin(), soil_lattice.end());
    cudaMemcpy(d_lattice, temp_lattice.data(), L * L * sizeof(char), cudaMemcpyHostToDevice);

    for (int step = 0; step < N_STEPS; ++step)
    {
        cudaMemset(d_latticeUpdated, 0, L * L * sizeof(bool));

        updateKernel<<<gridSize,
         blockSize>>>(d_lattice, d_latticeUpdated, p, d_state);
        cudaDeviceSynchronize();

        cudaMemcpy(d_lattice, d_latticeUpdated, L * L * sizeof(bool), cudaMemcpyDeviceToDevice);

        if (step >= RECORDING_STEP)
        {
            // Calculate distances and histogram
            cudaMemset(d_distances, -1, L * L * sizeof(int));
            cudaMemset(d_histogram, 0, L * sizeof(float));
            cudaMemset(d_nActiveSites, 0, sizeof(int));
            cudaMemset(d_maxIndex, 0, sizeof(int));

            dim3 gridSize(L, L);
            dim3 blockSize(1, 1);

            generateRandomNumbers<<<gridSize, blockSize>>>(d_lattice, d_randomNumbers, d_state);
            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error after generateRandomNumbers: %s\n", cudaGetErrorString(err));

            cudaMemcpyToSymbol(blk_num, &ZERO, sizeof(int));
            findMaxIndex<<<blocksPerGridCounting, threadsPerBlockCounting, 2 * threadsPerBlockCounting * sizeof(float) + 2 * threadsPerBlockCounting * sizeof(int)>>>(d_randomNumbers, d_maxIndex);
            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error after findMaxIndex: %s\n", cudaGetErrorString(err));

            calculateDistances<<<gridSize, blockSize>>>(d_lattice, d_distances, d_maxIndex);
            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error after calculateDistances: %s\n", cudaGetErrorString(err));

            calculateHistogram<<<gridSize, blockSize>>>(d_distances, d_histogram);
            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error after calculateHistogram: %s\n", cudaGetErrorString(err));

            countLattice<<<blocksPerGridCounting, threadsPerBlockCounting>>>(d_lattice, d_nActiveSites);
            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error after countLattice: %s\n", cudaGetErrorString(err));

            normalizeHistogram<<<L, 1>>>(d_histogram, d_nActiveSites);
            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error after normalizeHistogram: %s\n", cudaGetErrorString(err));

            addToTotalHistogram<<<L, 1>>>(d_histogram, d_histogram_total);
            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error after addToTotalHistogram: %s\n", cudaGetErrorString(err));
        }
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / (N_STEPS - 1) * 100 << "%\r" << std::flush;
    }

    std::vector<float> histogram(L);
    cudaMemcpy(histogram.data(), d_histogram_total, L * sizeof(float), cudaMemcpyDeviceToHost);

    // Normalize the histogram
    for (int i = 0; i < L; ++i)
    {
        if (i != 0)
        {
            histogram[i] /= i < L / 2 ? 4 * i : 4 * (L - i);
            histogram[i] /= (N_STEPS - RECORDING_STEP);
        }
    }

    std::cout << "Histogram: ";
    for (int i = 0; i < L; ++i)
    {
        std::cout << histogram[i] << " ";
    }

    for (int i = 0; i < L; ++i)
    {
        file << i << "\t" << histogram[i] << "\n";
    }

    cudaFree(d_lattice);
    cudaFree(d_latticeUpdated);
    cudaFree(d_state);
    cudaFree(d_distances);
    cudaFree(d_histogram);
    cudaFree(d_histogram_total);
    cudaFree(d_nActiveSites);
    cudaFree(d_randomNumbers);
    cudaFree(d_maxIndex);
}

int main(int argc, char *argv[])
{
    double p = P; // Use the default value P
    if (argc > 1) // If a command-line argument is provided
    {
        p = std::stod(argv[1]); // Convert the first argument to double and use it as p
    }

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/Corr2D/activePoint/p_" << p << "_L_" << L << ".tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    run(file, p); // Pass p to the run function
    file.close();

    return 0;
}