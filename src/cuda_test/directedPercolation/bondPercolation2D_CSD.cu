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
constexpr double P = 0.6;
constexpr int L = 1024;
constexpr int N_STEPS = 2000;
constexpr int RECORDING_STEP = N_STEPS / 2;
constexpr int RECORDING_INTERVAL = 20;

std::vector<bool> initLattice(int L)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    std::vector<bool> soil_lattice(L * L);
    for (int i = 0; i < L * L; ++i)
    {
        soil_lattice[i] = dis(gen);
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

__global__ void updateKernel(bool *d_lattice, bool *d_latticeUpdated, curandState *state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Initialize the RNG
    curandState localState = state[index]; // Copy the state to local memory for efficiency

    int nPercolationTrials = 0;

    if (d_lattice[idx + idy * L])
    {
        nPercolationTrials++;
    }

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

// ToDo: Warning!! `d_histogram` might overflow, change to `long long`
__global__ void cabaretKernel(bool *d_lattice, int *d_labelsFilled, int *d_labelsEmpty, int *d_histogramFilled, int *d_histogramEmpty, bool *d_changeOccurred)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Initialize the label for this pixel
    d_labelsFilled[index] = index;
    d_labelsEmpty[index] = index;

    // Perform the local labeling step for both filled and empty sites
    bool changeOccurred;
    do
    {
        // Reset the global flag at the start of each iteration
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            *d_changeOccurred = false;
        }

        __syncthreads();

        changeOccurred = false;
        for (int i = 0; i < 4; ++i)
        {
            int x = idx + (i % 2) * 2 - 1;
            int y = idy + (i / 2) * 2 - 1;

            // Periodic boundary conditions
            x = (x + L) % L;
            y = (y + L) % L;

            // Check if the site is occupied and has a lower label
            if (d_lattice[x + y * L] && d_labelsFilled[x + y * L] < d_labelsFilled[index])
            {
                d_labelsFilled[index] = d_labelsFilled[x + y * L];
                changeOccurred = true;
            }

            // Check if the site is empty and has a lower label
            if (!d_lattice[x + y * L] && d_labelsEmpty[x + y * L] < d_labelsEmpty[index])
            {
                d_labelsEmpty[index] = d_labelsEmpty[x + y * L];
                changeOccurred = true;
            }
        }

        // Update the global flag if a change occurred
        if (changeOccurred)
        {
            *d_changeOccurred = true;
        }

        // Wait for all threads to finish before checking the global flag
        __syncthreads();

    } while (*d_changeOccurred);

    // Perform the global labeling step for both filled and empty sites
    __syncthreads();
    if (d_labelsFilled[index] != index)
    {
        d_labelsFilled[index] = d_labelsFilled[d_labelsFilled[index]];
    }
    if (d_labelsEmpty[index] != index)
    {
        d_labelsEmpty[index] = d_labelsEmpty[d_labelsEmpty[index]];
    }

    // Calculate the cluster size for both filled and empty sites
    if (d_lattice[index]) {
        atomicAdd(&d_histogramFilled[d_labelsFilled[index]], 1);
    } else {
        atomicAdd(&d_histogramEmpty[d_labelsEmpty[index]], 1);
    }
}

void run(std::ofstream &file)
{
    std::vector<bool> soil_lattice = initLattice(L);

    cudaSetDevice(0);

    bool *d_lattice;
    bool *d_latticeUpdated;
    curandState *d_state;
    int *d_labelsFilled;
    int *d_labelsEmpty;
    int *d_histogramFilled;
    int *d_histogramEmpty;
    bool *d_changeOccurred;
    cudaMalloc(&d_lattice, L * L * sizeof(bool));
    cudaMalloc(&d_latticeUpdated, L * L * sizeof(bool));
    cudaMalloc(&d_state, L * L * sizeof(curandState));
    cudaMalloc(&d_labelsFilled, L * L * sizeof(int));
    cudaMalloc(&d_labelsEmpty, L * L * sizeof(int));
    cudaMalloc(&d_histogramFilled, L * L * sizeof(int));
    cudaMalloc(&d_histogramEmpty, L * L * sizeof(int));
    cudaMalloc(&d_changeOccurred, sizeof(bool));

    initCurand<<<L, L>>>(d_state, time(0));

    std::vector<char> temp_lattice(soil_lattice.begin(), soil_lattice.end());
    cudaMemcpy(d_lattice, temp_lattice.data(), L * L * sizeof(char), cudaMemcpyHostToDevice);

    dim3 blockSize(1, 1);
    dim3 gridSize(L, L);

    for (int step = 0; step < N_STEPS; ++step)
    {
        cudaMemset(d_latticeUpdated, 0, L * L * sizeof(bool));

        updateKernel<<<gridSize, blockSize>>>(d_lattice, d_latticeUpdated, d_state);
        cudaDeviceSynchronize();

        cudaMemcpy(d_lattice, d_latticeUpdated, L * L * sizeof(bool), cudaMemcpyDeviceToDevice);

        if (step >= RECORDING_STEP && step % RECORDING_INTERVAL == 0)
        {
            cudaMemset(d_labelsFilled, 0, L * L * sizeof(int));
            cudaMemset(d_labelsEmpty, 0, L * L * sizeof(int));
            cudaMemset(d_histogramFilled, 0, L * L * sizeof(int));
            cudaMemset(d_histogramEmpty, 0, L * L * sizeof(int));

            cabaretKernel<<<gridSize, blockSize>>>(d_lattice, d_labelsFilled, d_labelsEmpty, d_histogramFilled, d_histogramEmpty, d_changeOccurred);            cudaDeviceSynchronize();

            std::vector<int> histogramFilled(L * L);
            std::vector<int> histogramEmpty(L * L);
            cudaMemcpy(histogramFilled.data(), d_histogramFilled, L * L * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(histogramEmpty.data(), d_histogramEmpty, L * L * sizeof(int), cudaMemcpyDeviceToHost);

            std::stringstream ssFilled, ssEmpty;
            for (int i = 0; i < L * L; ++i)
            {
                if (histogramFilled[i] > 0) // Only write non-zero entries
                {
                    ssFilled << histogramFilled[i] << ",";
                }
                if (histogramEmpty[i] > 0) // Only write non-zero entries
                {
                    ssEmpty << histogramEmpty[i] << ",";
                }
            }

            std::string histogramStrFilled = ssFilled.str();
            histogramStrFilled = histogramStrFilled.substr(0, histogramStrFilled.length() - 1); // remove the last comma
            std::string histogramStrEmpty = ssEmpty.str();
            histogramStrEmpty = histogramStrEmpty.substr(0, histogramStrEmpty.length() - 1); // remove the last comma

            file << step << "\t" << histogramStrFilled << "\t" << histogramStrEmpty << "\n";
        }

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / (N_STEPS - 1) * 100 << "%\r" << std::flush;
    }

    cudaFree(d_lattice);
    cudaFree(d_latticeUpdated);
    cudaFree(d_state);
    cudaFree(d_labelsFilled);
    cudaFree(d_labelsEmpty);
    cudaFree(d_histogramFilled);
    cudaFree(d_histogramEmpty);
    cudaFree(d_changeOccurred);
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/CSD2D/p_" << P << "_L_" << L << ".tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "Step\tfilledClusterSizes\temptyClusterSizes\n";
    run(file);
    file.close();

    return 0;
}