#include <random>
#include <thread>
#include <algorithm>
#include <array>
#include <mutex>
#include <iostream>
#include <fstream>
#include <filesystem>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::mutex mtx;                                            // for progress output
int max_threads = std::thread::hardware_concurrency() - 2; // Keep 2 threads free
int active_threads = 0;
int completed_threads = 0;

std::random_device rd;
std::mt19937 gen(rd());

// Define constants
constexpr std::array<double, 6> p_values = {0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002};
constexpr int STEPS_PER_LATTICEPOINT = 2000;
constexpr int RECORDING_INTERVAL = 1;
constexpr int RECORDING_STEP = 0;
constexpr int N = 1000; // Define the size of your array

thread_local std::uniform_int_distribution<> dis(0, 1);
thread_local std::uniform_real_distribution<> dis_prob(0, 1);


std::vector<bool> initArray()
{
    std::vector<bool> array(N, false); // Initialize all elements to false
    for (int i = 0; i < N; ++i)
    {
        array[i] = dis(gen); // Randomly assign true or false
    }
    return array;
}

void updateArray(std::vector<bool> &array, double p)
{
    std::vector<bool> newArray(N, false); // Initialize a new array with all elements set to false

    for (int i = 0; i < N; ++i)
    {
        if (array[i]) // If the site is active
        {
            for (int j = 0; j < N; ++j) // Attempt to activate every site in the new array
            {
                if (dis_prob(gen) < p) // With probability p
                {
                    newArray[j] = true;
                }
            }
        }
    }

    array = newArray; // Set the array to the new array
}

void run(std::ofstream &file, double p)
{
    std::vector<bool> array = initArray();

    for (int step = 1; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        updateArray(array, p);

        if (step % RECORDING_INTERVAL == 0)
        {
            // Count the number of true sites
            int trueCount = std::count(array.begin(), array.end(), true);

            // Calculate the fraction of true sites
            double fraction = static_cast<double>(trueCount) / array.size();

            // Write the data for this step
            file << step << "\t" << fraction << "\n";
        }
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::thread> threads;

    for (double p : p_values)
    {
        threads.push_back(std::thread([p, &argv]() { // Capture p by value
            std::string exePath = argv[0];
            std::string exeDir = std::filesystem::path(exePath).parent_path().string();

            std::ostringstream filename;
            filename << exeDir << "/outputs/timeseriesWellMixed/p_" << p << "_N_" << N << ".tsv";

            std::ofstream file(filename.str());

            file << "step\tactive_fraction\n";

            run(file, p);

            file.close();

            // Lock the mutex before writing to the console
            std::lock_guard<std::mutex> lock(mtx);
            completed_threads++;
            std::cout << "Thread finished. Completion: " << (completed_threads * 100.0 / p_values.size()) << "%\n";
        }));
    }

    for (std::thread &t : threads)
    {
        t.join();
    }

    return 0;
}