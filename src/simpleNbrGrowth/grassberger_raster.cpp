#include <random>
#include <vector>
#include <thread>
#include <array>
#include <unordered_map>
#include <mutex>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::mutex mtx;                                            // for progress output
int max_threads = std::thread::hardware_concurrency() - 2; // Keep 2 threads free
int active_threads = 0;
int completed_threads = 0;

thread_local std::random_device rd;
thread_local std::mt19937 gen(rd());

// Define constants
constexpr double SIGMA = 1;
constexpr std::array<double, 8> theta_values = {0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66};
constexpr int L = 1024; // 2^10 = 1024
constexpr int STEPS_PER_LATTICEPOINT = 1000;
constexpr int N_SIMULATIONS = 10000;

std::uniform_int_distribution<> dis_site(0, L *L - 1);
std::uniform_int_distribution<> dis_dir(0, 3);
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<long long> createDistanceSquaredArray(int L)
{
    std::vector<long long> distanceSquared(L * L);

    int centerX = L / 2;
    int centerY = L / 2;

    for (int y = 0; y < L; ++y)
    {
        for (int x = 0; x < L; ++x)
        {
            int dx = x - centerX;
            int dy = y - centerY;
            distanceSquared[x + y * L] = dx * dx + dy * dy;
        }
    }

    return distanceSquared;
}

std::vector<bool> initLattice(int L)
{
    std::vector<bool> soil_lattice(L * L, false); // Initialize all sites to false
    int centralIndex = L * (L / 2) + L / 2;       // Calculate the index of the central site
    soil_lattice[centralIndex] = true;            // Set the central site to true
    return soil_lattice;
}

struct Coordinate
{
    int x;
    int y;
};

Coordinate get_random_neighbour(Coordinate site, int L)
{
    int dir = dis_dir(gen);
    switch (dir)
    {
    case 0: // left
        return {(site.x - 1 + L) % L, site.y};
    case 1: // right
        return {(site.x + 1) % L, site.y};
    case 2: // above
        return {site.x, (site.y - 1 + L) % L};
    case 3: // below
        return {site.x, (site.y + 1) % L};
    }
    return site; // should never reach here
}

void updateLattice(std::vector<bool> &lattice, double theta)
{
    // Choose a random site
    int site_index = dis_site(gen);
    Coordinate site = {site_index % L, site_index / L};

    if (lattice[site_index]) // if the site is occupied
    {
        // make it empty with rate theta
        if (dis_prob(gen) < theta)
        {
            lattice[site_index] = false;
        }
    }
    else // if the site is empty
    {
        // choose a random neighbour
        Coordinate nbr = get_random_neighbour(site, L);
        int nbr_index = nbr.y * L + nbr.x;

        // if the neighbour is occupied, make it occupied with rate SIGMA
        if (lattice[nbr_index] && dis_prob(gen) < SIGMA)
        {
            lattice[site_index] = true;
        }
    }
}

int countLattice(std::vector<bool> &lattice)
{
    int count = 0;
    for (bool site : lattice)
    {
        if (site)
        {
            count++;
        }
    }
    return count;
}

double calculateR2(std::vector<bool> &lattice, std::vector<long long> &distanceSquared, int activeSites)
{
    if (activeSites == 0)
    {
        return 0;
    }
    long long totalDistance = 0;
    for (int i = 0; i < L * L; ++i)
    {
        if (lattice[i])
        {
            totalDistance += distanceSquared[i];
        }
    }
    return static_cast<double>(totalDistance) / activeSites;
}

void run(std::ofstream &file, double theta)
{
    std::vector<long long> distanceSquared = createDistanceSquaredArray(L);

    for (int i = 0; i < N_SIMULATIONS; ++i)
    {
        std::vector<bool> lattice = initLattice(L);
        int activeSites = countLattice(lattice);
        file << i << "," << 0 << "," << activeSites << "," << calculateR2(lattice, distanceSquared, activeSites) << "\n";
        for (int step = 1; step <= STEPS_PER_LATTICEPOINT; ++step)
        {
            for (int i = 0; i < L * L; ++i)
            {
                updateLattice(lattice, theta);
            }

            int activeSites = countLattice(lattice);

            double r2 = calculateR2(lattice, distanceSquared, activeSites);

            // Write the data for this step
            file << i << "," << step << "," << activeSites << "," << r2 << "\n";

            if (activeSites == 0)
            {
                break;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::thread> threads;

    for (double theta : theta_values)
    {
        threads.push_back(std::thread([theta, &argv]() { // Capture theta by value
            std::vector<bool> lattice = initLattice(L);

            std::string exePath = argv[0];
            std::string exeDir = std::filesystem::path(exePath).parent_path().string();

            std::ostringstream filename;
            filename << exeDir << "/outputs/grassberger/sigma_" << SIGMA << "_theta_" << theta << ".csv";

            std::ofstream file(filename.str());

            file << "simulation,time,activeCounts,R2\n";

            run(file, theta);

            file.close();

            // Lock the mutex before writing to the console
            std::lock_guard<std::mutex> lock(mtx);
            completed_threads++;
            std::cout << "Thread finished. Completion: " << (completed_threads * 100.0 / theta_values.size()) << "%\n";
        }));
    }

    for (std::thread &t : threads)
    {
        t.join();
    }

    return 0;
}