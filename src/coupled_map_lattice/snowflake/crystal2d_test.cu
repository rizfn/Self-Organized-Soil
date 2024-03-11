#include <cuda.h>     // for CUDA-related functions
#include <iostream>   // for std::cout and std::flush
#include <filesystem> // for std::filesystem::path
#include <fstream>    // for std::ofstream
#include <sstream>    // for std::ostringstream

constexpr float Tc = 1.0f;
constexpr float T0 = -0.5f;
constexpr int L = 100;
constexpr int N_STEPS = 1000;
constexpr float D = 0.2f;
constexpr float C1 = 0.3f;
constexpr float C2 = 0.95f;

// Initialize the data arrays
void init(float *u1, float *u2, float *x1, float *x2)
{
    for (int i = 0; i < L * L; i++)
    {
        u1[i] = T0;
        x1[i] = 0.0;
    }
    u1[L / 2 * L + L / 2] = Tc;
    x1[L / 2 * L + L / 2] = 1.0;
}

// Diffusion kernel
__global__ void diffuse(float *u1, float *u2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1)
    {
        u2[i * L + j] = u1[i * L + j] + D * (u1[(i + 1) * L + j] + u1[(i - 1) * L + j] + u1[i * L + j + 1] + u1[i * L + j - 1] - 4 * u1[i * L + j]);
    }
}

// Update kernel
__global__ void update(float *u1, float *u2, float *x1, float *x2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1)
    {
        if (u1[i * L + j] < Tc && (x1[(i + 1) * L + j] >= 1.0 || x1[i * L + j + 1] >= 1.0 || x1[(i - 1) * L + j] >= 1.0 || x1[i * L + j - 1] >= 1.0))
        {
            x2[i * L + j] = x1[i * L + j] + C1 * (Tc - u1[i * L + j]);
            u2[i * L + j] = u1[i * L + j] + C2 * (Tc - u1[i * L + j]);
        }
        else
        {
            x2[i * L + j] = x1[i * L + j];
            u2[i * L + j] = u1[i * L + j];
        }
    }
}

__global__ void resetCenter(float *u)
{
    int index = L / 2 * L + L / 2;
    u[index] = Tc;
}

void run(std::ofstream &file)
{
    // Define the data arrays
    float u1[L * L];
    float u2[L * L];
    float x1[L * L];
    float x2[L * L];

    init(u1, u2, x1, x2);

    // Copy the data to the GPU
    float *d_u1;
    float *d_u2;
    float *d_x1;
    float *d_x2;
    cudaMalloc(&d_u1, L * L * sizeof(float));
    cudaMalloc(&d_u2, L * L * sizeof(float));
    cudaMalloc(&d_x1, L * L * sizeof(float));
    cudaMalloc(&d_x2, L * L * sizeof(float));
    cudaMemcpy(d_u1, u1, L * L * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2, u2, L * L * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x1, x1, L * L * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, x2, L * L * sizeof(float), cudaMemcpyHostToDevice);

    // // Define the block and grid sizes
    dim3 blockSize(1, 1);
    dim3 gridSize(L, L);

    // Run the update function on the GPU
    for (int step = 0; step < N_STEPS; step++)
    {
        diffuse<<<gridSize, blockSize>>>(d_u1, d_u2);
        cudaDeviceSynchronize(); // wait for diffuse to finish

        resetCenter<<<1, 1>>>(d_u2);
        cudaDeviceSynchronize(); // wait for resetCenter to finish

        // Swap the u1 and u2 pointers
        float *temp = d_u1;
        d_u1 = d_u2;
        d_u2 = temp;

        update<<<gridSize, blockSize>>>(d_u1, d_u2, d_x1, d_x2);
        cudaDeviceSynchronize(); // wait for update to finish

        resetCenter<<<1, 1>>>(d_u2);
        cudaDeviceSynchronize(); // wait for resetCenter to finish

        // Swap the u1 and u2 pointers
        temp = d_u1;
        d_u1 = d_u2;
        d_u2 = temp;

        // Swap the x1 and x2 pointers
        temp = d_x1;
        d_x1 = d_x2;
        d_x2 = temp;

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / (N_STEPS - 1) * 100 << "%\r" << std::flush;
    }

    // Copy the data back to the CPU
    cudaMemcpy(u1, d_u1, L * L * sizeof(float), cudaMemcpyDeviceToHost);
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
    cudaFree(d_u1);
    cudaFree(d_u2);
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