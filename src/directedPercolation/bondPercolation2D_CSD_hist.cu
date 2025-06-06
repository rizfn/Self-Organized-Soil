// CUDA C++
#include <cuda.h>
#include <curand_kernel.h>
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

// Define constants
constexpr double P = 0.3178;
constexpr int L = 4096 * 2;
constexpr int N_STEPS = 2000;
constexpr int RECORDING_STEP = N_STEPS / 2;
constexpr int RECORDING_INTERVAL = 5;

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
            // Only initialize "odd" cells
            if ((i + j) % 2 == 1)
            {
                soil_lattice[i * L + j] = dis(gen);
            }
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
        if (curand_uniform(&localState) < 1 - pow(1 - p, nPercolationTrials))
        {
            d_latticeUpdated[idx + idy * L] = true;
        }
    }

    // Update the state
    state[index] = localState;
}

class UnionFind
{
public:
    UnionFind(int n) : parent(n), rank(n, 0)
    {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int i)
    {
        if (parent[i] != i)
            parent[i] = find(parent[i]);
        return parent[i];
    }

    void union_set(int i, int j)
    {
        int ri = find(i), rj = find(j);
        if (ri != rj)
        {
            if (rank[ri] < rank[rj])
                parent[ri] = rj;
            else if (rank[ri] > rank[rj])
                parent[rj] = ri;
            else
            {
                parent[ri] = rj;
                ++rank[rj];
            }
        }
    }

private:
    std::vector<int> parent, rank;
};

std::pair<std::vector<int>, std::vector<int>> get_cluster_sizes(const std::vector<bool> &lattice)
{
    int newL = L / 2;
    std::vector<bool> condensed_lattice(L * newL, false);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < newL; ++j)
        {
            condensed_lattice[i * newL + j] = lattice[i * L + j * 2] || lattice[i * L + j * 2 + 1];
        }
    }

    UnionFind uf_filled(L * newL);
    UnionFind uf_empty(L * newL);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < newL; ++j)
        {
            if (condensed_lattice[i * newL + j])
            {
                if (condensed_lattice[((i - 1 + L) % L) * newL + j])
                    uf_filled.union_set(i * newL + j, ((i - 1 + L) % L) * newL + j);
                if (condensed_lattice[i * newL + ((j - 1 + newL) % newL)])
                    uf_filled.union_set(i * newL + j, i * newL + ((j - 1 + newL) % newL));
                if (condensed_lattice[((i + 1) % L) * newL + j])
                    uf_filled.union_set(i * newL + j, ((i + 1) % L) * newL + j);
                if (condensed_lattice[i * newL + ((j + 1) % newL)])
                    uf_filled.union_set(i * newL + j, i * newL + ((j + 1) % newL));
            }
            else
            {
                if (!condensed_lattice[((i - 1 + L) % L) * newL + j])
                    uf_empty.union_set(i * newL + j, ((i - 1 + L) % L) * newL + j);
                if (!condensed_lattice[i * newL + ((j - 1 + newL) % newL)])
                    uf_empty.union_set(i * newL + j, i * newL + ((j - 1 + newL) % newL));
                if (!condensed_lattice[((i + 1) % L) * newL + j])
                    uf_empty.union_set(i * newL + j, ((i + 1) % L) * newL + j);
                if (!condensed_lattice[i * newL + ((j + 1) % newL)])
                    uf_empty.union_set(i * newL + j, i * newL + ((j + 1) % newL));
            }
        }
    }

    std::unordered_map<int, int> cluster_sizes_filled;
    std::unordered_map<int, int> cluster_sizes_empty;
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < newL; ++j)
        {
            if (condensed_lattice[i * newL + j])
            {
                int root = uf_filled.find(i * newL + j);
                ++cluster_sizes_filled[root];
            }
            else
            {
                int root = uf_empty.find(i * newL + j);
                ++cluster_sizes_empty[root];
            }
        }
    }

    std::vector<int> sizes_filled;
    for (const auto &pair : cluster_sizes_filled)
        sizes_filled.push_back(pair.second);

    std::vector<int> sizes_empty;
    for (const auto &pair : cluster_sizes_empty)
        sizes_empty.push_back(pair.second);

    return {sizes_filled, sizes_empty};
}

void run(std::ofstream &file, double p)
{
    std::vector<bool> soil_lattice = initLattice(L);

    cudaSetDevice(0);

    bool *d_lattice;
    bool *d_latticeUpdated;
    curandState *d_state;
    cudaMalloc(&d_lattice, L * L * sizeof(bool));
    cudaMalloc(&d_latticeUpdated, L * L * sizeof(bool));
    cudaMalloc(&d_state, L * L * sizeof(curandState));

    dim3 blockSize(1, 1);
    dim3 gridSize(L, L);

    initCurand<<<gridSize, blockSize>>>(d_state, time(0));

    cudaError_t curandError = cudaGetLastError();
    if (curandError != cudaSuccess)
    {
        std::cerr << "CUDA error in initCurand: " << cudaGetErrorString(curandError) << std::endl;
    }

    std::vector<char> temp_lattice(soil_lattice.begin(), soil_lattice.end());
    cudaMemcpy(d_lattice, temp_lattice.data(), L * L * sizeof(char), cudaMemcpyHostToDevice);

    for (int step = 0; step < N_STEPS; ++step)
    {
        cudaError_t updateError = cudaGetLastError();
        if (updateError != cudaSuccess)
        {
            std::cerr << "CUDA error in updateKernel: " << cudaGetErrorString(updateError) << std::endl;
        }

        cudaMemset(d_latticeUpdated, 0, L * L * sizeof(bool));

        updateKernel<<<gridSize, blockSize>>>(d_lattice, d_latticeUpdated, p, d_state);
        cudaDeviceSynchronize();

        cudaMemcpy(d_lattice, d_latticeUpdated, L * L * sizeof(bool), cudaMemcpyDeviceToDevice);

        if (step >= RECORDING_STEP && step % RECORDING_INTERVAL == 0)
        {

            // Copy lattice data from GPU to CPU
            std::vector<char> lattice_cpu(L * L);
            cudaMemcpy(lattice_cpu.data(), d_lattice, L * L * sizeof(char), cudaMemcpyDeviceToHost);
            std::vector<bool> lattice_bool(lattice_cpu.begin(), lattice_cpu.end());

            // Calculate cluster sizes
            auto [sizes_filled, sizes_empty] = get_cluster_sizes(lattice_bool);

            file << step << "\t";
            // Write cluster sizes to a file
            for (size_t i = 0; i < sizes_filled.size(); ++i)
            {
                file << sizes_filled[i];
                if (i != sizes_filled.size() - 1)
                    file << ",";
            }
            file << "\t";
            for (size_t i = 0; i < sizes_empty.size(); ++i)
            {
                file << sizes_empty[i];
                if (i != sizes_empty.size() - 1)
                    file << ",";
            }
            file << "\n";
        }

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / (N_STEPS - 1) * 100 << "%\r" << std::flush;
    }

    cudaFree(d_lattice);
    cudaFree(d_latticeUpdated);
    cudaFree(d_state);
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
    // filePathStream << exeDir << "/outputs/CSD2D/criticalPoints/p_" << p << "_L_" << L << ".tsv";
    // filePathStream << exeDir << "/outputs/CSD2D/otherPoints/p_" << p << "_L_" << L << ".tsv";
    // filePathStream << exeDir << "/outputs/CSD2D/criticalPointsLarge/tests/p_" << p << "_L_" << L << ".tsv";
    filePathStream << exeDir << "/outputs/CSD2D/criticalPointsVLarge/p_" << p << "_L_" << L << ".tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "Step\tfilledClusterSizes\temptyClusterSizes\n";
    run(file, p); // Pass p to the run function
    file.close();

    return 0;
}