#include <random>
#include <vector>
#include <thread>
#include <array>
#include <unordered_map>
#include <algorithm>
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

std::random_device rd;
std::mt19937 gen(rd());

// Define constants
constexpr std::array<double, 15> p_values = {0.34, 0.341, 0.342, 0.3422, 0.3424, 0.3426, 0.3428, 0.343, 0.3432, 0.3434, 0.3436, 0.3438, 0.344, 0.345, 0.346};
constexpr int L = 1024; 
constexpr int STEPS_PER_LATTICEPOINT = 4000;
constexpr int RECORDING_INTERVAL = 2;
constexpr int RECORDING_STEP = STEPS_PER_LATTICEPOINT / 2;

thread_local std::uniform_int_distribution<> dis(0, 1);
thread_local std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<bool> initLattice()
{
    std::vector<bool> soil_lattice(L * L, false); // Initialize all cells to false
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if ((i + j) % 2 == 1) // Set odd cells to true
            {
                soil_lattice[i * L + j] = dis(gen);
            }
        }
    }
    return soil_lattice;
}

struct Coordinate
{
    int x;
    int y;
};

void updateLattice(std::vector<bool> &lattice, double p)
{
    std::vector<bool> newLattice(L * L, false);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            int index = i * L + j;
            int nPercolationTrials = 0;

            for (int k = 0; k < 4; ++k)
            {
                int x = i + (k % 2);
                int y = j + (k / 2);

                // Periodic boundary conditions
                x = (x + L) % L;
                y = (y + L) % L;

                // Check if the site is occupied
                if (lattice[x * L + y])
                {
                    nPercolationTrials++;
                }
            }

            if (nPercolationTrials > 0)
            {
                if (dis_prob(gen) < 1 - pow(1 - p, nPercolationTrials))
                {
                    newLattice[index] = true;
                }
            }
        }
    }
    lattice = newLattice;
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

std::pair<bool, bool> does_largest_cluster_span(const std::vector<bool> &lattice)
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
    int spanning_cluster_root = -1;
    for (const auto &pair : cluster_sizes_filled)
    {
        if (pair.second > max_cluster_size)
        {
            max_cluster_size = pair.second;
            spanning_cluster_root = pair.first;
        }
    }

    if (spanning_cluster_root == -1)
        return std::make_pair(false, false);

    // Check if the largest cluster spans the lattice
    std::vector<bool> x_coords(L, false);
    std::vector<bool> y_coords(L, false);

    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            int index = i * L + j;
            if (uf_filled.find(index) == spanning_cluster_root)
            {
                x_coords[i] = true;
                y_coords[j] = true;
            }
        }
    }

    bool spans_x = std::all_of(x_coords.begin(), x_coords.end(), [](bool v)
                               { return v; });
    bool spans_y = std::all_of(y_coords.begin(), y_coords.end(), [](bool v)
                               { return v; });

    return std::make_pair(spans_x, spans_y);
}

void run(std::ofstream &file, double p)
{
    std::vector<bool> lattice = initLattice();

    for (int step = 1; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        updateLattice(lattice, p);

        if (step % (RECORDING_INTERVAL) == 0)
        {
            if (step >= RECORDING_STEP)
            {
                std::pair<bool, bool> spanning_clusters = does_largest_cluster_span(lattice);

                // Write the data for this step
                file << step << "\t" << spanning_clusters.first << "\t" << spanning_clusters.second << "\n";
            }
        }
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::thread> threads;

    for (double p : p_values)
    {
        threads.push_back(std::thread([p, &argv]() { // Capture p by value
            std::vector<bool> lattice = initLattice();

            std::string exePath = argv[0];
            std::string exeDir = std::filesystem::path(exePath).parent_path().string();

            std::ostringstream filename;
            filename << exeDir << "/outputs/orderParameter/L_" << L << "/p_" << p << ".tsv";

            std::ofstream file(filename.str());

            file << "step\tx_span\ty_span\n";

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