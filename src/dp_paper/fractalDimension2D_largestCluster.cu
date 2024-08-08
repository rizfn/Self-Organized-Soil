// CUDA C++
#include <cuda.h>
#include <curand_kernel.h>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include <numeric>
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
constexpr double P = 0.34375; // 0.344 for FSPL, 0.318 for ESPL
constexpr int L = 4096;
constexpr int N_STEPS = 4000;
constexpr int RECORDING_STEP = N_STEPS * 5 / 10;
constexpr int RECORDING_SKIP = 5;
template<int N>
constexpr int log2()
{
    return (N < 2) ? 0 : 1 + log2<N / 2>();
}
constexpr int N_BOX_SIZES = log2<L>() - 1;


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

void boxCount2D(std::vector<bool> &lattice, std::vector<unsigned int> &boxCounts)
{
    unsigned int s = 2;
    unsigned int size = L;
    unsigned char ni = 0;

    while (size > 2)
    {
        int sm = s >> 1; // s/2
        unsigned long im;
        unsigned long ismm;

        boxCounts[ni] = 0;
        for (unsigned long i = 0; i < (L - 1); i += s)
        {
            im = i * L;
            ismm = (i + sm) * L;
            for (unsigned long j = 0; j < (L - 1); j += s)
            {
                lattice[im + j] = lattice[im + j] || lattice[im + (j + sm)] || lattice[ismm + j] || lattice[ismm + (j + sm)];
                boxCounts[ni] += lattice[im + j];
            }
        }
        ni++;
        s <<= 1;    // s *= 2;
        size >>= 1; // size /= 2;
    }
}

std::vector<unsigned int> get_fractal_dimension(const std::vector<bool> &lattice)
{
    UnionFind uf_filled(L * L);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            int index = i * L + j;
            if (lattice[index])
            {
                if (lattice[((i - 1 + L) % L) * L + j])
                    uf_filled.union_set(index, ((i - 1 + L) % L) * L + j);
                if (lattice[i * L + (j - 1 + L) % L])
                    uf_filled.union_set(index, i * L + (j - 1 + L) % L);
                if (lattice[((i + 1) % L) * L + j])
                    uf_filled.union_set(index, ((i + 1) % L) * L + j);
                if (lattice[i * L + (j + 1) % L])
                    uf_filled.union_set(index, i * L + (j + 1) % L);
            }
        }
    }

    std::unordered_map<int, int> cluster_sizes_filled;
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            int index = i * L + j;
            if (lattice[index])
            {
                int root = uf_filled.find(index);
                ++cluster_sizes_filled[root];
            }
        }
    }

    int max_cluster_size = 0;
    int max_cluster_root = -1;
    for (const auto &pair : cluster_sizes_filled)
    {
        if (pair.second > max_cluster_size)
        {
            max_cluster_size = pair.second;
            max_cluster_root = pair.first;
        }
    }

    std::vector<bool> max_cluster_mask(L * L, false);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            int index = i * L + j;
            if (lattice[index] && uf_filled.find(index) == max_cluster_root)
            {
                max_cluster_mask[index] = true;
            }
        }
    }

    std::vector<unsigned int> box_counts(N_BOX_SIZES, 0);
    boxCount2D(max_cluster_mask, box_counts);

    return box_counts;

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

    dim3 blockSize(1, 1);
    dim3 gridSize(L, L);

    initCurand<<<gridSize, blockSize>>>(d_state, time(0));

    std::vector<char> temp_lattice(soil_lattice.begin(), soil_lattice.end());
    cudaMemcpy(d_lattice, temp_lattice.data(), L * L * sizeof(char), cudaMemcpyHostToDevice);

    std::vector<unsigned int> box_lengths(N_BOX_SIZES);
    box_lengths[0] = 2;
    for (size_t i = 1; i < N_BOX_SIZES; ++i) {
        box_lengths[i] = box_lengths[i - 1] * 2;
    }

    for (int step = 0; step < N_STEPS; ++step)
    {
        cudaMemset(d_latticeUpdated, 0, L * L * sizeof(bool));

        updateKernel<<<gridSize, blockSize>>>(d_lattice, d_latticeUpdated, p, d_state);
        cudaDeviceSynchronize();

        cudaMemcpy(d_lattice, d_latticeUpdated, L * L * sizeof(bool), cudaMemcpyDeviceToDevice);

        if ((step >= RECORDING_STEP) && (step % RECORDING_SKIP == 0))
        {
            // copy lattice to CPU
            std::vector<char> lattice_cpu(L * L);
            cudaMemcpy(lattice_cpu.data(), d_lattice, L * L * sizeof(char), cudaMemcpyDeviceToHost);
            std::vector<bool> lattice_bool(lattice_cpu.begin(), lattice_cpu.end());

            std::vector<unsigned int> box_counts = get_fractal_dimension(lattice_bool);
            
            for (size_t i = 0; i < N_BOX_SIZES; ++i)
            {
                file << step << "\t" << box_lengths[i] << "\t" << box_counts[i] << std::endl;
            }
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
    filePathStream << exeDir << "/outputs/FractalDim/p_" << p << "_L_" << L << ".tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "step\tbox_length\tbox_count" << std::endl;
    run(file, p); // Pass p to the run function
    file.close();

    return 0;
}