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
constexpr double SIGMA = 0.5;
constexpr double THETA = 0.03;
constexpr double MU = 1;
constexpr double RHO1 = 0.25;
constexpr double RHO2 = 1;
constexpr int L = 512; // 2^10 = 1024
constexpr int N_STEPS = 10000;

constexpr int BLOCK_LENGTH = 2;

constexpr int EMPTY = 0;
constexpr int NUTRIENT = 1;
constexpr int SOIL = 2;
constexpr int GREEN = 3; // host
constexpr int BLUE = 4;  // parasite

std::vector<int> initLattice(int L)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_site(0, 4);

    std::vector<int> soil_lattice(L * L);
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

__global__ void countLattice(int *d_lattice, int *d_counts, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        int value = d_lattice[idx];
        atomicAdd(&d_counts[value], 1);
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

__global__ void updateKernel(int *d_lattice, curandState *state, int offsetX, int offsetY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Initialize the RNG
    curandState localState = state[index]; // Copy the state to local memory for efficiency

    int squareX = (blockIdx.x * BLOCK_LENGTH + offsetX * BLOCK_LENGTH / 2 + threadIdx.x) % L;
    int squareY = (blockIdx.y * BLOCK_LENGTH + offsetY * BLOCK_LENGTH / 2 + threadIdx.y) % L;

    // Select a random site in the 2x2 square (for block length 4)
    int siteX = squareX + curand(&localState) % BLOCK_LENGTH / 2;
    int siteY = squareY + curand(&localState) % BLOCK_LENGTH / 2;

    // Get the value at the selected site
    int siteValue = d_lattice[siteX * L + siteY];

    // Perform the update
    if (siteValue == EMPTY || siteValue == NUTRIENT)
    {
        // empty or nutrient
        // choose a random neighbour
        int nbrX, nbrY;
        getRandomNeighbor(siteX, siteY, L, &localState, &nbrX, &nbrY);
        int nbrValue = d_lattice[nbrX * L + nbrY];

        if (nbrValue == SOIL)
        {
            // if neighbour is soil
            // fill with soil-filling rate
            if (curand_uniform(&localState) < SIGMA)
            {
                d_lattice[siteX * L + siteY] = SOIL;
            }
        }
    }
    else if (siteValue == GREEN)
    {
        // check for death
        if (curand_uniform(&localState) < THETA)
        {
            d_lattice[siteX * L + siteY] = EMPTY;
        }
        else
        {
            // move into a neighbour
            int new_siteX, new_siteY;
            getRandomNeighbor(siteX, siteY, L, &localState, &new_siteX, &new_siteY);
            int new_siteValue = d_lattice[new_siteX * L + new_siteY];

            // move the worm
            d_lattice[new_siteX * L + new_siteY] = GREEN;
            d_lattice[siteX * L + siteY] = EMPTY;

            // check if the new site is a nutrient that this worm can consume
            if (new_siteValue == NUTRIENT)
            {
                // reproduce behind you
                if (curand_uniform(&localState) < RHO1)
                {
                    d_lattice[siteX * L + siteY] = GREEN;
                }
            }
            // check if the new site is soil
            else if (new_siteValue == SOIL)
            {
                // leave nutrient behind
                if (curand_uniform(&localState) < MU)
                {
                    d_lattice[siteX * L + siteY] = NUTRIENT;
                }
            }
            // check if the new site is a worm
            else if (new_siteValue == BLUE || new_siteValue == GREEN)
            {
                // keep both with worms (undo the vacant space in original site)
                d_lattice[siteX * L + siteY] = new_siteValue;
            }
        }
    }
    else if (siteValue == BLUE)
    {
        // check for death
        if (curand_uniform(&localState) < THETA)
        {
            d_lattice[siteX * L + siteY] = EMPTY;
        }
        else
        {
            // move into a neighbour
            int new_siteX, new_siteY;
            getRandomNeighbor(siteX, siteY, L, &localState, &new_siteX, &new_siteY);
            int new_siteValue = d_lattice[new_siteX * L + new_siteY];

            // move the worm
            d_lattice[new_siteX * L + new_siteY] = BLUE;
            d_lattice[siteX * L + siteY] = EMPTY;

            // check if the new site is a nutrient that this worm can consume
            if (new_siteValue == NUTRIENT)
            {
                // reproduce behind you
                if (curand_uniform(&localState) < RHO2)
                {
                    d_lattice[siteX * L + siteY] = BLUE;
                }
            }
            // check if the new site is a worm
            else if (new_siteValue == BLUE || new_siteValue == GREEN)
            {
                // keep both with worms (undo the vacant space in original site)
                d_lattice[siteX * L + siteY] = new_siteValue;
            }
        }
    }

    // Update the state
    state[index] = localState;
}

void run(std::ofstream &file)
{
    // Initialize the lattice
    std::vector<int> soil_lattice = initLattice(L);

    // Initialize CUDA
    cudaSetDevice(0);

    // Allocate memory on the GPU
    int *d_lattice;
    curandState *d_state;
    cudaMalloc(&d_lattice, L * L * sizeof(int));
    cudaMalloc(&d_state, L * L * sizeof(curandState));

    // Initialize the RNG states
    initCurand<<<L / BLOCK_LENGTH, L / BLOCK_LENGTH>>>(d_state, time(0));

    // Copy the lattice data to the GPU
    cudaMemcpy(d_lattice, soil_lattice.data(), L * L * sizeof(int), cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    dim3 blockSize(1, 1);
    dim3 gridSize(L / BLOCK_LENGTH, L / BLOCK_LENGTH);

    // Initialize counts to 0
    int *d_countArray;
    cudaMalloc(&d_countArray, N_STEPS * 5 * sizeof(int));
    cudaMemset(d_countArray, 0, N_STEPS * 5 * sizeof(int));

    int threadsPerBlockCounting = 256;
    int blocksPerGridCounting = (L * L + threadsPerBlockCounting - 1) / threadsPerBlockCounting;

    // Launch the CUDA kernel for each of the A, B, C, and D squares
    for (int step = 0; step < N_STEPS; ++step)
    {
        for (int i = 0; i < BLOCK_LENGTH / 2 * BLOCK_LENGTH / 2; ++i) // 1 iteration per square in subblock
        {
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 0, 0); // A squares
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 1, 0); // B squares
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 0, 1); // C squares
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 1, 1); // D squares
            cudaDeviceSynchronize();
        }

        countLattice<<<blocksPerGridCounting, threadsPerBlockCounting>>>(d_lattice, d_countArray + step * 5, L * L);

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / (N_STEPS - 1) * 100 << "%\r" << std::flush;
    }

    float *d_normalizedArray;
    cudaMalloc(&d_normalizedArray, N_STEPS * 5 * sizeof(float));

    normalizeCounts<<<blocksPerGridCounting, threadsPerBlockCounting>>>(d_countArray, d_normalizedArray);
    cudaDeviceSynchronize();

    std::vector<float> counts(N_STEPS * 5);
    cudaMemcpy(counts.data(), d_normalizedArray, N_STEPS * 5 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_STEPS * 5; i += 5)
    {
        for (int j = 0; j < 5; ++j)
        {
            file << counts[i + j];
            if (j < 4) // Don't add a comma after the last entry on a line
                file << ",";
        }
        file << "\n";
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
    // filePathStream << exeDir << "/outputs/timeseries/L_" << L << "_sigma_" << SIGMA << "_theta_" << THETA << "_rhofactor_" << RHO2 / RHO1 << ".csv";
    // filePathStream << exeDir << "/outputs/timeseries/L_" << L << "_sigma_" << SIGMA << "_theta_" << THETA << "_rhofactor_" << RHO2 / RHO1 << "_bl_" << BLOCK_LENGTH << ".csv";
    filePathStream << exeDir << "/outputs/timeseries/bl_sigma_" << SIGMA << "_theta_" << THETA << "_rhofactor_" << RHO2 / RHO1 << "/L_" << L << "_bl_" << BLOCK_LENGTH << ".csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "empty,nutrient,soil,green,blue\n";
    run(file);
    file.close();

    return 0;
}