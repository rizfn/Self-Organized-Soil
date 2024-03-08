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
constexpr double THETA = 0.05;
constexpr double RHO = 1;
constexpr double MU = 1;
constexpr int L = 100; 
constexpr int N_STEPS = 10000;
constexpr int RECORDING_STEP = N_STEPS - 100;

constexpr int calculateBlockLength(int L)
{
    int tempL = L;
    int power = 0;
    while (tempL > 4096)
    {                // up to 4096, bL=4. 8192, bL=8. 16384, bL=16, etc.
        tempL >>= 1; // equivalent to tempL = tempL / 2;
        power++;
    }
    return (1 << power) < 4 ? 4 : (1 << power); // equivalent to return pow(2, power);
}
constexpr int BLOCK_LENGTH = 4;

constexpr int N = 4; // number of species
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
constexpr std::array<int, N> WORMS = []
{
    std::array<int, N> arr{};
    for (int i = 0; i < N; ++i)
    {
        arr[i] = i + N + 1;
    }
    return arr;
}();
__constant__ int d_WORMS[N];
__constant__ int d_NUTRIENTS[N];

std::vector<int> initLattice(int L)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_site(0, 2 * N);

    std::vector<int> soil_lattice(L * L * L);
    for (int i = 0; i < L * L * L; ++i)
    {
        soil_lattice[i] = dis_site(gen);
    }
    return soil_lattice;
}

__global__ void initCurand(curandState *state, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x + idz * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    // Initialize the RNG state for this thread
    curand_init(seed, index, 0, &state[index]);
}

__device__ void getRandomNeighbor(int x, int y, int z, int L, curandState *localState, int *nbr_x, int *nbr_y, int *nbr_z)
{
    float rand = curand_uniform(localState);
    if (rand < 1.0f/3)
    {
        *nbr_x = (x + (curand_uniform(localState) < 0.5f ? -1 : 1) + L) % L;
        *nbr_y = y;
        *nbr_z = z;
    }
    else if (rand < 2.0f/3)
    {
        *nbr_x = x;
        *nbr_y = (y + (curand_uniform(localState) < 0.5f ? -1 : 1) + L) % L;
        *nbr_z = z;
    }
    else
    {
        *nbr_x = x;
        *nbr_y = y;
        *nbr_z = (z + (curand_uniform(localState) < 0.5f ? -1 : 1) + L) % L;
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

__global__ void updateKernel(int *d_lattice, curandState *state, int offsetX, int offsetY, int offsetZ)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x + idz * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    // Initialize the RNG
    curandState localState = state[index]; // Copy the state to local memory for efficiency

    int cube_x = (blockIdx.x * BLOCK_LENGTH + offsetX * BLOCK_LENGTH / 2 + threadIdx.x) % L;
    int cube_y = (blockIdx.y * BLOCK_LENGTH + offsetY * BLOCK_LENGTH / 2 + threadIdx.y) % L;
    int cube_z = (blockIdx.z * BLOCK_LENGTH + offsetZ * BLOCK_LENGTH / 2 + threadIdx.z) % L;

    // Select a random site in the (bl/2 x bl/2 x bl/2) cube
    int site_x = cube_x + curand(&localState) % BLOCK_LENGTH / 2;
    int site_y = cube_y + curand(&localState) % BLOCK_LENGTH / 2;
    int site_z = cube_z + curand(&localState) % BLOCK_LENGTH / 2;

    // Get the value at the selected site
    int site_value = d_lattice[site_x * L * L + site_y * L + site_z];

    for (int i = 0; i < N; ++i)
    {
        if (site_value == d_WORMS[i])
        {
            // worm of species i
            // check for death
            if (curand_uniform(&localState) < THETA)
            {
                d_lattice[site_x * L * L + site_y * L + site_z] = EMPTY;
            }
            else
            {
                // move into a neighbour
                int new_site_x, new_site_y, new_site_z;
                getRandomNeighbor(site_x, site_y, site_z, L, &localState, &new_site_x, &new_site_y, &new_site_z);
                int new_site_value = d_lattice[new_site_x * L * L + new_site_y * L + new_site_z];

                // move the worm
                d_lattice[new_site_x * L * L + new_site_y * L + new_site_z] = d_WORMS[i];
                d_lattice[site_x * L * L + site_y * L + site_z] = EMPTY;

                // check if the new site is a nutrient that this worm can consume
                if (new_site_value == d_NUTRIENTS[(i + 1) % N])
                {
                    // reproduce behind you
                    if (curand_uniform(&localState) < RHO)
                    {
                        d_lattice[site_x * L * L + site_y * L + site_z] = d_WORMS[i];
                    }
                }
                // check if the new site is empty
                else if (new_site_value == EMPTY)
                {
                    // leave nutrient behind
                    if (curand_uniform(&localState) < MU)
                    {
                        d_lattice[site_x * L * L + site_y * L + site_z] = d_NUTRIENTS[i];
                    }
                }
                // check if the new site is a worm
                else if (findInArray(d_WORMS, N, new_site_value))
                {
                    // keep both with worms (undo the vacant space in original site)
                    d_lattice[site_x * L * L + site_y * L + site_z] = new_site_value;
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
    cudaMalloc(&d_lattice, L * L * L * sizeof(int));
    cudaMalloc(&d_state, L * L * L * sizeof(curandState));

    // Initialize the RNG states
    initCurand<<<dim3(L / BLOCK_LENGTH, L / BLOCK_LENGTH, L / BLOCK_LENGTH), dim3(BLOCK_LENGTH, BLOCK_LENGTH, BLOCK_LENGTH)>>>(d_state, time(0));

    // Copy the lattice data to the GPU
    cudaMemcpy(d_lattice, soil_lattice.data(), L * L * L * sizeof(int), cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    dim3 blockSize(1, 1, 1);
    dim3 gridSize(L / BLOCK_LENGTH, L / BLOCK_LENGTH, L / BLOCK_LENGTH);

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
        for (int i = 0; i < BLOCK_LENGTH / 2 * BLOCK_LENGTH / 2 * BLOCK_LENGTH / 2; ++i) // 1 iteration per cube in subblock
        {
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 0, 0, 0); // A cubes
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 1, 0, 0); // B cubes
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 0, 1, 0); // C cubes
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 1, 1, 0); // D cubes
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 0, 0, 1); // E cubes
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 1, 0, 1); // F cubes
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 0, 1, 1); // G cubes
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_lattice, d_state, 1, 1, 1); // H cubes
            cudaDeviceSynchronize();
        }
        if (step >= RECORDING_STEP)
        {
            // Copy the updated lattice data back to the CPU
            cudaMemcpy(soil_lattice.data(), d_lattice, L * L * L * sizeof(int), cudaMemcpyDeviceToHost);

            file << step << "\t["; // Use \t as separator
            for (int matrix_index = 0; matrix_index < L; ++matrix_index)
            {
                file << "[";
                for (int row_index = 0; row_index < L; ++row_index)
                {
                    file << "[";
                    for (int cell_index = 0; cell_index < L; ++cell_index)
                    {
                        file << soil_lattice[matrix_index * L * L + row_index * L + cell_index];
                        if (cell_index != L - 1) // Check if it's not the last element in the row
                        {
                            file << ",";
                        }
                    }
                    file << "]";
                    if (row_index != L - 1) // Check if it's not the last row
                    {
                        file << ",";
                    }
                }
                file << "]";
                if (matrix_index != L - 1) // Check if it's not the last matrix
                {
                    file << ",";
                }
            }
            file << "]";
            if (step != N_STEPS) // Check if it's not the last step
            {
                file << "\n";
            }
        }
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / N_STEPS * 100 << "%\r" << std::flush;
    }

    // Copy the updated lattice data back to the CPU
    cudaMemcpy(soil_lattice.data(), d_lattice, L * L * L * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the memory allocated on the GPU
    cudaFree(d_lattice);
    cudaFree(d_state);
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "\\..\\..\\..\\docs\\data\\multispec_nutrient\\" << N << "spec\\nosoil_theta_" << THETA << ".tsv";
    std::string filePath = filePathStream.str();
    
    std::ofstream file;
    file.open(filePath);
    file << "step\tlattice\n";
    run(N_STEPS, file);
    file.close();

    return 0;
}