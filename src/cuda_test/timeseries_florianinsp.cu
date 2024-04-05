// CUDA C++
#include <cuda.h>
#include <curand_kernel.h>
#include <random>
#include <vector>
#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

// Define constants
constexpr double SIGMA = 1;
constexpr double THETA = 0.5;
constexpr int L = 1024; // 2^10 = 1024
constexpr int N_STEPS = 5000;

constexpr int BLOCK_LENGTH = 16;

std::vector<bool> initLattice(int L)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_site(0, 1);

    std::vector<bool> soil_lattice(L * L);
    for (int i = 0; i < L * L; ++i)
    {
        soil_lattice[i] = dis_site(gen);
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

__device__ void getRandomNeighbor(int x, int y, int L, curandState *localState, int *nbrX, int *nbrY)
{
    if (curand_uniform(localState) < 0.5f)
    {
        *nbrX = (x + (curand_uniform(localState) < 0.5f ? -1 : 1) + L) % L;
        *nbrY = y;
    }
    else
    {
        *nbrX = x;
        *nbrY = (y + (curand_uniform(localState) < 0.5f ? -1 : 1) + L) % L;
    }
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

__global__ void updateKernel(bool *d_lattice, curandState *state, int blockType)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Initialize the RNG
    curandState localState = state[index]; // Copy the state to local memory for efficiency

    // Check if this cell should be updated in this step
    if ((blockIdx.x + blockIdx.y) % 4 == blockType)
    {
        // Generate random coordinates within the block
        int siteX = blockIdx.x * BLOCK_LENGTH + curand(&localState) % BLOCK_LENGTH;
        int siteY = blockIdx.y * BLOCK_LENGTH + curand(&localState) % BLOCK_LENGTH;
        
        // Get the value at the selected site
        bool siteValue = d_lattice[siteX * L + siteY];

        // Perform the update
        if (siteValue) // if the site is occupied
        {
            // make it empty with rate THETA
            if (curand_uniform(&localState) < THETA)
            {
                d_lattice[siteX * L + siteY] = false;
            }
        }
        else // if the site is empty
        {
            // choose a random neighbour of the selected site
            int nbrX, nbrY;
            getRandomNeighbor(siteX, siteY, L, &localState, &nbrX, &nbrY);
            bool nbrValue = d_lattice[nbrX * L + nbrY];

            // if the neighbour is occupied, make it occupied with rate SIGMA
            if (nbrValue && curand_uniform(&localState) < SIGMA)
            {
                d_lattice[siteX * L + siteY] = true;
            }
        }
    }

    // Update the state
    state[index] = localState;
}

void run(std::ofstream &file)
{
    // Initialize the lattice
    std::vector<bool> soil_lattice = initLattice(L);

    // Initialize CUDA
    cudaSetDevice(0);

    // Allocate memory on the GPU
    bool *d_lattice;
    curandState *d_state;
    cudaMalloc(&d_lattice, L * L * sizeof(bool));
    cudaMalloc(&d_state, L * L * sizeof(curandState));

    // Initialize the RNG states
    initCurand<<<L / BLOCK_LENGTH, L / BLOCK_LENGTH>>>(d_state, time(0));

    // Copy the lattice data to a temporary std::vector<char> for the cudaMemcpy call
    std::vector<char> temp_lattice(soil_lattice.begin(), soil_lattice.end());

    // Copy the lattice data to the GPU
    cudaMemcpy(d_lattice, temp_lattice.data(), L * L * sizeof(char), cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    dim3 blockSize(1, 1);
    dim3 gridSize(L / BLOCK_LENGTH, L / BLOCK_LENGTH);

    // Initialize counts to 0
    int *d_countArray;
    cudaMalloc(&d_countArray, N_STEPS * sizeof(int));
    cudaMemset(d_countArray, 0, N_STEPS * sizeof(int));

    int threadsPerBlockCounting = 256;
    int blocksPerGridCounting = (L * L + threadsPerBlockCounting - 1) / threadsPerBlockCounting;

    // Launch the CUDA kernel for each of the A, B, C, and D squares
    for (int step = 0; step < N_STEPS; ++step)
    {
        for (int i = 0; i < BLOCK_LENGTH * BLOCK_LENGTH; ++i) // 1 iteration per square in block
        {
            // Update 'A' cells
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 0);
            cudaDeviceSynchronize();

            // Update 'B' cells
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 1);
            cudaDeviceSynchronize();

            // Update 'C' cells
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 2);
            cudaDeviceSynchronize();

            // Update 'D' cells
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 3);
            cudaDeviceSynchronize();
        }

        countLattice<<<blocksPerGridCounting, threadsPerBlockCounting>>>(d_lattice, d_countArray + step, L * L);

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

    // Free the memory allocated on the GPU
    cudaFree(d_lattice);
    cudaFree(d_state);
    cudaFree(d_countArray);
    cudaFree(d_normalizedArray);
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    // filePathStream << exeDir << "/outputs/sigma_" << SIGMA << "_theta_" << THETA << "/L_" << L << "_bl_" << BLOCK_LENGTH << ".csv";
    filePathStream << exeDir << "/outputs/florian/L_" << L << "_bl_" << BLOCK_LENGTH << "_insp.csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    run(file);
    file.close();

    return 0;
}