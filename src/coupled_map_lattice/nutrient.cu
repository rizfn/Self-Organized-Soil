#include <cuda.h>     // for CUDA-related functions
#include <iostream>   // for std::cout and std::flush
#include <filesystem> // for std::filesystem::path
#include <fstream>    // for std::ofstream
#include <sstream>    // for std::ostringstream

constexpr int L = 500;
constexpr int N_STEPS = 2000;
constexpr float D = 0.2;
constexpr float rho = 1;
constexpr float sigma = 1;
constexpr float theta = 1;
constexpr float soil0 = 0.25;
constexpr float worm0 = 0.25;
constexpr float nutrient0 = 0.25;
constexpr float empty0 = 1 - soil0 - worm0 - nutrient0;

// Initialize the data arrays
__global__ void init(float *s1, float *s2, float *w1, float *w2, float *n1, float *n2, float *e1, float *e2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i * L + j;

    if (i < L && j < L)
    {
        // Initialize soil with a constant density
        s1[index] = soil0;
        s2[index] = soil0;

        // Initialize worms with constant density
        w1[index] = worm0;
        w2[index] = worm0;

        // Initialize nutrients with a constant density
        n1[index] = nutrient0;
        n2[index] = nutrient0;

        // Initialize empty spaces with a constant value
        e1[index] = empty0;
        e2[index] = empty0;
    }
}

__global__ void update(float *s1, float *s2, float *w1, float *w2, float *n1, float *n2, float *e1, float *e2, float theta, float rho, float sigma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i * L + j;

    // Calculate the indices of the neighboring sites with periodic boundary conditions
    int iPlus1 = (i + 1) % L;
    int iMinus1 = (i - 1 + L) % L;
    int jPlus1 = (j + 1) % L;
    int jMinus1 = (j - 1 + L) % L;

    // Kill worms at a constant rate theta and add to empty state
    float killedWorms = w1[index] * theta;
    w1[index] -= killedWorms;
    e1[index] += killedWorms;

    // Calculate the total amount of worms that diffuse from the current site to each of its neighbors
    float diffusedWorms = D * (w1[iPlus1 * L + j] + w1[iMinus1 * L + j] + w1[i * L + jPlus1] + w1[i * L + jMinus1] - 4 * w1[index]);

    // Calculate the backflows from each neighbor
    float wormFlow = diffusedWorms / 4;
    float backflowNutrients = wormFlow * (s1[iPlus1 * L + j] + s1[iMinus1 * L + j] + s1[i * L + jPlus1] + s1[i * L + jMinus1]);
    float backflowWorms = rho * wormFlow * (n1[iPlus1 * L + j] + n1[iMinus1 * L + j] + n1[i * L + jPlus1] + n1[i * L + jMinus1]) + wormFlow * (w1[iPlus1 * L + j] + w1[iMinus1 * L + j] + w1[i * L + jPlus1] + w1[i * L + jMinus1]);
    // float backflowEmpty = wormFlow * (e1[iPlus1 * L + j] + e1[iMinus1 * L + j] + e1[i * L + jPlus1] + e1[i * L + jMinus1]) + (1 - rho) * wormFlow * (n1[iPlus1 * L + j] + n1[iMinus1 * L + j] + n1[i * L + jPlus1] + n1[i * L + jMinus1]);
    float backflowEmpty = diffusedWorms - backflowNutrients - backflowWorms;

    // Update the states based on the backflows
    n2[index] = n1[index] + backflowNutrients;
    w2[index] = w1[index] + backflowWorms;
    e2[index] = e1[index] + backflowEmpty;

    // Update soil, empty state, and nutrients based on sigma S (E+N)
    s2[index] += sigma * s1[index] * (e1[index] + n1[index]); // Soil increases with sigma S (E+N)
    e2[index] -= sigma * s1[index] * e1[index]; // Empty state decreases with sigma S E
    n2[index] -= sigma * s1[index] * n1[index]; // Nutrients decrease with sigma S N
}

__global__ void copy(float *src, float *dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < L && j < L)
    {
        dst[i * L + j] = src[i * L + j];
    }
}


void run(std::ofstream &file)
{
    // Copy the data to the GPU
    float *d_s1, *d_s2; // soil density
    float *d_w1, *d_w2; // worm density
    float *d_n1, *d_n2; // nutrient density
    float *d_e1, *d_e2; // empty state
    cudaMalloc(&d_s1, L * L * sizeof(float));
    cudaMalloc(&d_s2, L * L * sizeof(float));
    cudaMalloc(&d_w1, L * L * sizeof(float));
    cudaMalloc(&d_w2, L * L * sizeof(float));
    cudaMalloc(&d_n1, L * L * sizeof(float));
    cudaMalloc(&d_n2, L * L * sizeof(float));
    cudaMalloc(&d_e1, L * L * sizeof(float));
    cudaMalloc(&d_e2, L * L * sizeof(float));

    // Define the block and grid sizes
    dim3 blockSize(1, 1);
    dim3 gridSize(L, L);

    init<<<gridSize, blockSize>>>(d_s1, d_s2, d_w1, d_w2, d_n1, d_n2, d_e1, d_e2);
    cudaDeviceSynchronize(); // wait for init to finish

    // Run the update function on the GPU
    for (int step = 0; step < N_STEPS; step++)
    {
        update<<<gridSize, blockSize>>>(d_s1, d_s2, d_w1, d_w2, d_n1, d_n2, d_e1, d_e2, theta, rho, sigma);
        cudaDeviceSynchronize(); // wait for update to finish

        copy<<<gridSize, blockSize>>>(d_s2, d_s1);
        copy<<<gridSize, blockSize>>>(d_w2, d_w1);
        copy<<<gridSize, blockSize>>>(d_n2, d_n1);
        copy<<<gridSize, blockSize>>>(d_e2, d_e1);
        cudaDeviceSynchronize(); // wait for copy to finish

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / (N_STEPS - 1) * 100 << "%\r" << std::flush;
    }

    // Copy the data back to the CPU
    float *s1 = new float[L * L];
    float *w1 = new float[L * L];
    float *n1 = new float[L * L];
    float *e1 = new float[L * L];
    cudaMemcpy(s1, d_s1, L * L * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w1, d_w1, L * L * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(n1, d_n1, L * L * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(e1, d_e1, L * L * sizeof(float), cudaMemcpyDeviceToHost);

    // Write the final lattices to the file
    for (int i = 0; i < L; i++)
    {
        for (int j = 0; j < L; j++)
        {
            file << s1[i * L + j] << "," << w1[i * L + j] << "," << n1[i * L + j] << "," << e1[i * L + j];
            if (j < L - 1)
            {
                file << ",";
            }
        }
        file << "\n";
    }

    // Free the GPU memory
    cudaFree(d_s1);
    cudaFree(d_s2);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_n1);
    cudaFree(d_n2);
    cudaFree(d_e1);
    cudaFree(d_e2);
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/test.csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    run(file);
    file.close();

    return 0;
}