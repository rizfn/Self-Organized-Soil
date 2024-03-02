// CUDA C++
#include <cuda.h>
#include <curand_kernel.h>
#include <random>
#include <vector>
#include <array>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

// Define constants
constexpr double SIGMA = 0.5;
constexpr double THETA = 0.015;
constexpr double RHO = 1;
constexpr double MU = 1;
constexpr int L = 4096; // 2^10 = 1024
constexpr int N_STEPS = 10000;

constexpr int calculateBlockLength(int L) {
    int tempL = L;
    int power = 0;
    while (tempL > 4096) {  // up to 4096, bL=4. 8192, bL=8. 16384, bL=16, etc.
        tempL >>= 1; // equivalent to tempL = tempL / 2;
        power++;
    }
    return (1 << power) < 4 ? 4 : (1 << power); // equivalent to return pow(2, power);
}
// constexpr int blockLength = calculateBlockLength(L);  // TODO: Something seems off. For blocklengths of 8, I get larger clusters?
constexpr int blockLength = 4;

constexpr int N = 5; // number of species
constexpr int EMPTY = 0;
constexpr std::array<int, N> NUTRIENTS = []
{
    std::array<int, N> arr{};
    for (int i = 0; i < N; ++i)
    {
        arr[i] = i + 1;
    }
    return arr;
}();
constexpr int SOIL = N + 1;
constexpr std::array<int, N> WORMS = []
{
    std::array<int, N> arr{};
    for (int i = 0; i < N; ++i)
    {
        arr[i] = i + N + 2;
    }
    return arr;
}();
__constant__ int d_WORMS[N];
__constant__ int d_NUTRIENTS[N];

std::vector<int> initLattice(int L)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_site(0, 2 * N + 1);

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

__device__ void getRandomNeighbor(int x, int y, int L, curandState *localState, int *nbr_x, int *nbr_y)
{
    if (curand_uniform(localState) < 0.5f)
    {
        *nbr_x = (x + (curand_uniform(localState) < 0.5f ? -1 : 1) + L) % L;
        *nbr_y = y;
    }
    else
    {
        *nbr_x = x;
        *nbr_y = (y + (curand_uniform(localState) < 0.5f ? -1 : 1) + L) % L;
    }
}

__device__ bool findInArray(const int *arr, int size, int value)
{
    for (int i = 0; i < size; ++i)
    {
        if (arr[i] == value)
        {
            return true;
        }
    }
    return false;
}

__global__ void updateKernel(int *d_lattice, curandState *state, int offsetX, int offsetY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Initialize the RNG
    curandState localState = state[index]; // Copy the state to local memory for efficiency

    int square_x = (blockIdx.x * blockLength + offsetX * blockLength/2 + threadIdx.x) % L;
    int square_y = (blockIdx.y * blockLength + offsetY * blockLength/2 + threadIdx.y) % L;

    // Select a random site in the 4x4 square
    int site_x = square_x + curand(&localState) % blockLength/2;
    int site_y = square_y + curand(&localState) % blockLength/2;

    // Get the value at the selected site
    int site_value = d_lattice[site_x * L + site_y];

    // Perform the update
    if (site_value == EMPTY || findInArray(d_NUTRIENTS, N, site_value))
    {
        // empty or nutrient
        // choose a random neighbour
        int nbr_x, nbr_y;
        getRandomNeighbor(site_x, site_y, L, &localState, &nbr_x, &nbr_y);
        int nbr_value = d_lattice[nbr_x * L + nbr_y];

        if (nbr_value == SOIL)
        {
            // if neighbour is soil
            // fill with soil-filling rate
            if (curand_uniform(&localState) < SIGMA)
            {
                d_lattice[site_x * L + site_y] = SOIL;
            }
        }
    }
    else
    {
        for (int i = 0; i < N; ++i)
        {
            if (site_value == d_WORMS[i])
            {
                // worm of species i
                // check for death
                if (curand_uniform(&localState) < THETA)
                {
                    d_lattice[site_x * L + site_y] = EMPTY;
                }
                else
                {
                    // move into a neighbour
                    int new_site_x, new_site_y;
                    getRandomNeighbor(site_x, site_y, L, &localState, &new_site_x, &new_site_y);
                    int new_site_value = d_lattice[new_site_x * L + new_site_y];

                    // move the worm
                    d_lattice[new_site_x * L + new_site_y] = d_WORMS[i];
                    d_lattice[site_x * L + site_y] = EMPTY;

                    // check if the new site is a nutrient that this worm can consume
                    if (new_site_value == d_NUTRIENTS[(i + 1) % N])
                    {
                        // reproduce behind you
                        if (curand_uniform(&localState) < RHO)
                        {
                            d_lattice[site_x * L + site_y] = d_WORMS[i];
                        }
                    }
                    // check if the new site is soil
                    else if (new_site_value == SOIL)
                    {
                        // leave nutrient behind
                        if (curand_uniform(&localState) < MU)
                        {
                            d_lattice[site_x * L + site_y] = d_NUTRIENTS[i];
                        }
                    }
                    // check if the new site is a worm
                    else if (findInArray(d_WORMS, N, new_site_value))
                    {
                        // keep both with worms (undo the vacant space in original site)
                        d_lattice[site_x * L + site_y] = new_site_value;
                    }
                }
            }
        }
    }

    // Update the state
    state[index] = localState;
}

void run(int N_STEPS, std::ofstream &file)
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
    initCurand<<<L / blockLength, L / blockLength>>>(d_state, time(0));

    // Copy the lattice data to the GPU
    cudaMemcpy(d_lattice, soil_lattice.data(), L * L * sizeof(int), cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    dim3 blockSize(1, 1);
    dim3 gridSize(L / blockLength, L / blockLength);

    // Copy the WORMS and NUTRIENTS data to the constant memory on the GPU
    cudaMemcpyToSymbol(d_WORMS, WORMS.data(), N * sizeof(int));
    cudaMemcpyToSymbol(d_NUTRIENTS, NUTRIENTS.data(), N * sizeof(int));

    // print last cuda error
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
    }

    // Launch the CUDA kernel for each of the A, B, C, and D squares
    for (int step = 1; step <= N_STEPS; ++step)
    {
        for (int i = 0; i < blockLength/2 * blockLength/2; ++i)  // 1 teration per square in subblock
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

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / N_STEPS * 100 << "%\r" << std::flush;
    }

    // Copy the updated lattice data back to the CPU
    cudaMemcpy(soil_lattice.data(), d_lattice, L * L * sizeof(int), cudaMemcpyDeviceToHost);

    // Write the final state of the lattice to the file
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            file << soil_lattice[i * L + j] << ' ';
        }
        file << '\n';
    }

    // Free the memory allocated on the GPU
    cudaFree(d_lattice);
    cudaFree(d_state);
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "/../outputs/lattice2D/" << N << "spec/GPU_sigma_" << SIGMA << "_theta_" << THETA << ".tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    run(N_STEPS, file);
    file.close();

    return 0;
}