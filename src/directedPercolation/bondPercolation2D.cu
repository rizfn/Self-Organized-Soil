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

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);

std::vector<bool> initLattice()
{
    std::vector<bool> soil_lattice(L * L, false); // Initialize all sites to false
    int centralIndex = L * (L / 2) + L / 2;       // Calculate the index of the central site
    soil_lattice[centralIndex] = true;            // Set the central site to true
    return soil_lattice;
}

__global__ void initCurand(curandState *state, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Use a hash function to calculate the seed
    unsigned long long hash = index;
    hash = (hash ^ (hash >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    hash = (hash ^ (hash >> 27)) * UINT64_C(0x94d049bb133111eb);
    hash = hash ^ (hash >> 31);

    // Initialize the RNG state for this thread
    curand_init(seed + hash, index, 0, &state[index]);
}

__global__ void updateKernel(bool *d_lattice, bool *d_latticeUpdated, curandState *state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Initialize the RNG
    curandState localState = state[index]; // Copy the state to local memory for efficiency

    // Define the directions for the neighbors
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};

    if (d_lattice[idx + idy * L]) // if the site is active
    {
        for (int i = 0; i < 4; ++i)
        {
            int x = (idx + dx[i] + L) % L;
            int y = (idy + dy[i] + L) % L;

            if (curand_uniform(&localState) < P)
            {
                d_latticeUpdated[x + y * L] = true;
            }
        }
    }

    // Update the state
    state[index] = localState;
}

void run(std::ofstream &file)
{
    // Initialize the lattice
    std::vector<bool> soil_lattice = initLattice();

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

    // Launch the CUDA kernel for each of the A, B, C, and D squares
    for (int step = 0; step < N_STEPS; ++step)
    {
        // reset the updated lattice to all zeros
        cudaMemset(d_latticeUpdated, 0, L * L * sizeof(bool));

        updateKernel<<<gridSize, blockSize>>>(d_lattice, d_latticeUpdated, d_state);
        cudaDeviceSynchronize();

        cudaMemcpy(d_lattice, d_latticeUpdated, L * L * sizeof(bool), cudaMemcpyDeviceToDevice);

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / (N_STEPS - 1) * 100 << "%\r" << std::flush;
    }

    // Copy the lattice data back to the CPU
    cudaMemcpy(temp_lattice.data(), d_lattice, L * L * sizeof(char), cudaMemcpyDeviceToHost);

    // Write the lattice to the file
    for (int i = 0; i < L * L; ++i)
    {
        file << static_cast<int>(temp_lattice[i]);
        if ((i + 1) % L == 0)
        {
            file << std::endl;
        }
        else
        {
            file << ",";
        }
    }

    // Free the memory on the GPU
    cudaFree(d_lattice);
    cudaFree(d_latticeUpdated);
    cudaFree(d_state);
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/lattice2D/CUDA_p_" << P << "_L_" << L << ".csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    run(file);
    file.close();

    return 0;
}