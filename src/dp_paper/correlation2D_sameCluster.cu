// CUDA C++
#include <cuda.h>
#include <curand_kernel.h>
#include <chrono>
#include <random>
#include <vector>
#include <stack>
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
constexpr double P = 0.344; // 0.344 for FSPL, 0.318 for ESPL
constexpr int L = 4096;
constexpr int N_STEPS = 4000;
constexpr int RECORDING_STEP = N_STEPS * 5 / 10;

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

void dfs(const std::vector<bool> &lattice, std::vector<std::vector<bool>> &visited, std::vector<std::vector<bool>> &mask, int x, int y, int &size)
{
    std::stack<std::pair<int, int>> stack;
    stack.push({x, y});

    while (!stack.empty())
    {
        auto [cx, cy] = stack.top();
        stack.pop();

        if (visited[cx][cy])
            continue;

        visited[cx][cy] = true; // Mark the current site as visited
        mask[cx][cy] = true;    // Mark the current site in the mask
        size++;                 // Increment the size of the cluster

        // Directions: up, down, left, right
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};

        for (int i = 0; i < 4; ++i)
        {                                                             // Explore all four directions
            int nx = (cx + dx[i] + L) % L, ny = (cy + dy[i] + L) % L; // Periodic boundary conditions
            if (!visited[nx][ny] && lattice[nx * L + ny] == lattice[cx * L + cy])
            {
                stack.push({nx, ny}); // Push the neighbor to the stack if it is part of the cluster and not visited
            }
        }
    }
}

// Function to get the size of the cluster that includes the targetPoint, considering periodic boundary conditions
std::pair<int, std::vector<std::vector<bool>>> get_cluster_size(const std::vector<bool> &lattice, std::pair<int, int> targetPoint)
{
    std::vector<std::vector<bool>> visited(L, std::vector<bool>(L, false));
    std::vector<std::vector<bool>> mask(L, std::vector<bool>(L, false));
    int size = 0;                                                             // Initialize cluster size
    dfs(lattice, visited, mask, targetPoint.first, targetPoint.second, size); // Start DFS from targetPoint
    return {size, mask};                                                      // Return the size of the cluster and the mask
}

void run(std::ofstream &file, double p)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, L * L - 1);

    std::vector<bool> soil_lattice = initLattice(L);

    cudaSetDevice(0);

    bool *d_lattice;
    bool *d_latticeUpdated;
    curandState *d_state;
    cudaMalloc(&d_lattice, L * L * sizeof(bool));
    cudaMalloc(&d_latticeUpdated, L * L * sizeof(bool));
    cudaMalloc(&d_state, L * L * sizeof(curandState));

    std::vector<float> total_histogram(L, 0.0f);

    dim3 blockSize(1, 1);
    dim3 gridSize(L, L);

    initCurand<<<gridSize, blockSize>>>(d_state, time(0));

    std::vector<char> temp_lattice(soil_lattice.begin(), soil_lattice.end());
    cudaMemcpy(d_lattice, temp_lattice.data(), L * L * sizeof(char), cudaMemcpyHostToDevice);

    for (int step = 0; step < N_STEPS; ++step)
    {
        cudaMemset(d_latticeUpdated, 0, L * L * sizeof(bool));

        updateKernel<<<gridSize, blockSize>>>(d_lattice, d_latticeUpdated, p, d_state);
        cudaDeviceSynchronize();

        cudaError err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error after updateKernel: %s\n", cudaGetErrorString(err));

        cudaMemcpy(d_lattice, d_latticeUpdated, L * L * sizeof(bool), cudaMemcpyDeviceToDevice);

        if (step >= RECORDING_STEP)
        {
            // copy lattice to CPU
            std::vector<char> lattice_cpu(L * L);
            cudaMemcpy(lattice_cpu.data(), d_lattice, L * L * sizeof(char), cudaMemcpyDeviceToHost);
            std::vector<bool> lattice_bool(lattice_cpu.begin(), lattice_cpu.end());

            // randomly choose a point on the lattice which is true
            int activePoint = dis(gen);
            while (!lattice_bool[activePoint])
            {
                activePoint = dis(gen);
            }

            // get the cluster size and mask
            auto [clusterSize, mask] = get_cluster_size(lattice_bool, {activePoint / L, activePoint % L});

            if (clusterSize == 1)
            {
                continue;
            }

            std::vector<int> distances(clusterSize);
            int distanceArrayCounter = 0;
            for (int i = 0; i < L; ++i)
            {
                for (int j = 0; j < L; ++j)
                {
                    if (mask[i][j] == 1)
                    {
                        int dx = abs(activePoint / L - i);
                        int dy = abs(activePoint % L - j);
                        dx = std::min(dx, L - dx); // Account for periodic boundary conditions
                        dy = std::min(dy, L - dy); // Account for periodic boundary conditions
                        distances[distanceArrayCounter] = dx + dy;
                        distanceArrayCounter++;
                    }
                }
            }

            // for (int i = 0; i < distanceArrayCounter; ++i)
            // {
            //     std::cout << distances[i] << " ";
            // }
            // std::cout << std::endl;

            std::vector<float> histogram(L, 0.0f);
            for (int i = 0; i < distanceArrayCounter; ++i)
            {
                histogram[distances[i]] += 1.0f;
            }

            // for (int i = 0; i < L; ++i)
            // {
            //     std::cout << histogram[i] << " ";
            // }
            // std::cout << std::endl;

            for (int i = 0; i < L; ++i)
            {
                histogram[i] /= ((float)clusterSize - 1) / ((float)L * L);
                total_histogram[i] += histogram[i];
            }
            // for (int i = 0; i < L; ++i)
            // {
            //     std::cout << histogram[i] << " ";
            // }
            // std::cout << std::endl;
        }
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / (N_STEPS - 1) * 100 << "%\r" << std::flush;
    }

    // Normalize the histogram
    for (int i = 0; i < L; ++i)
    {
        if (i != 0)
        {
            total_histogram[i] /= i < L / 2 ? 4 * i : 4 * (L - i);
            total_histogram[i] /= (N_STEPS - RECORDING_STEP);
        }
    }

    std::cout << "Histogram: ";
    for (int i = 0; i < L; ++i)
    {
        std::cout << total_histogram[i] << " ";
    }

    for (int i = 0; i < L; ++i)
    {
        file << i << "\t" << total_histogram[i] << "\n";
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
    filePathStream << exeDir << "/outputs/Corr2D/sameCluster/p_" << p << "_L_" << L << ".tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    run(file, p); // Pass p to the run function
    file.close();

    return 0;
}