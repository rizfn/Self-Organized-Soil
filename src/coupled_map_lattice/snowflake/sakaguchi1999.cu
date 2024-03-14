#include <cuda.h>     // for CUDA-related functions
#include <iostream>   // for std::cout and std::flush
#include <filesystem> // for std::filesystem::path
#include <fstream>    // for std::ofstream
#include <sstream>    // for std::ostringstream

constexpr float Tc = 1;
constexpr float T0 = -0.6;
constexpr int L = 500;
constexpr int N_STEPS = 2000;
constexpr float D = 0.2;
constexpr float C1 = 0.075;
constexpr float C2 = 0.25;
constexpr float g = 0;

// Initialize the data arrays
__global__ void init(float *t1, float *t2, float *x1, float *x2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i * L + j;

    if (i < L && j < L)
    {
        t1[index] = T0;
        x1[index] = 0.0;
        if (i == L / 2 && j == L / 2)
        {
            t1[index] = Tc;
            x1[index] = 1.0;
        }
    }
}

// Diffusion kernel
__global__ void diffuse(float *t1, float *t2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1)
    {
        t2[i * L + j] = t1[i * L + j] + D * (t1[(i + 1) * L + j] + t1[(i - 1) * L + j] + t1[i * L + j + 1] + t1[i * L + j - 1] - 4 * t1[i * L + j]);
    }
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

__global__ void update(float *t1, float *t2, float *x1, float *x2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1)
    {
        // Calculate Noc
        int Noc = (x1[(i + 1) * L + j] >= 1.0) + (x1[i * L + j + 1] >= 1.0) + (x1[(i - 1) * L + j] >= 1.0) + (x1[i * L + j - 1] >= 1.0) +
                  (x1[(i + 1) * L + j + 1] >= 1.0) + (x1[(i + 1) * L + j - 1] >= 1.0) + (x1[(i - 1) * L + j + 1] >= 1.0) + (x1[(i - 1) * L + j - 1] >= 1.0);

        // Calculate Tc(i; j)
        float Tc_ij = Tc + g * (Noc - 3);

        if (t1[i * L + j] < Tc_ij && (x1[(i + 1) * L + j] >= 1.0 || x1[i * L + j + 1] >= 1.0 || x1[(i - 1) * L + j] >= 1.0 || x1[i * L + j - 1] >= 1.0))
        {
            x2[i * L + j] = x1[i * L + j] + C1 * (Tc_ij - t1[i * L + j]);
            t2[i * L + j] = t1[i * L + j] + C2 * (Tc_ij - t1[i * L + j]);
        }
        else
        {
            x2[i * L + j] = x1[i * L + j];
            t2[i * L + j] = t1[i * L + j];
        }
    }
    else if (i == 0 || i == L - 1 || j == 0 || j == L - 1)
    {
        x2[i * L + j] = x1[i * L + j];
        t2[i * L + j] = t1[i * L + j];
    }
}

void run(std::ofstream &file)
{
    // Copy the data to the GPU
    float *d_t1;
    float *d_t2;
    float *d_x1;
    float *d_x2;
    cudaMalloc(&d_t1, L * L * sizeof(float));
    cudaMalloc(&d_t2, L * L * sizeof(float));
    cudaMalloc(&d_x1, L * L * sizeof(float));
    cudaMalloc(&d_x2, L * L * sizeof(float));

    // Define the block and grid sizes
    dim3 blockSize(1, 1);
    dim3 gridSize(L, L);

    init<<<gridSize, blockSize>>>(d_t1, d_t2, d_x1, d_x2);
    cudaDeviceSynchronize(); // wait for init to finish

    // Run the update function on the GPU
    for (int step = 0; step < N_STEPS; step++)
    {
        diffuse<<<gridSize, blockSize>>>(d_t1, d_t2);
        cudaDeviceSynchronize(); // wait for diffuse to finish

        copy<<<gridSize, blockSize>>>(d_t2, d_t1);
        cudaDeviceSynchronize(); // wait for copy to finish

        update<<<gridSize, blockSize>>>(d_t1, d_t2, d_x1, d_x2);
        cudaDeviceSynchronize(); // wait for update to finish

        copy<<<gridSize, blockSize>>>(d_t2, d_t1);
        copy<<<gridSize, blockSize>>>(d_x2, d_x1);
        cudaDeviceSynchronize(); // wait for copy to finish

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / (N_STEPS - 1) * 100 << "%\r" << std::flush;
    }

    // Copy the data back to the CPU
    float *t1 = new float[L * L];
    float *x1 = new float[L * L];
    cudaMemcpy(t1, d_t1, L * L * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x1, d_x1, L * L * sizeof(float), cudaMemcpyDeviceToHost);

    // Write the x array to the file
    for (int i = 0; i < L; i++)
    {
        for (int j = 0; j < L; j++)
        {
            file << x1[i * L + j];
            if (j < L - 1)
            {
                file << ",";
            }
        }
        file << "\n";
    }

    // Free the GPU memory
    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_x1);
    cudaFree(d_x2);
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/Tc_" << Tc << "_T0_" << T0 << "_L_" << L << "_D_" << D << "_C1_" << C1 << "_C2_" << C2 << ".csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    run(file);
    file.close();

    return 0;
}