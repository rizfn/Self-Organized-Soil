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

std::mutex mtx;  // for progress output
int max_threads = std::thread::hardware_concurrency() - 2; // Keep 2 threads free
int active_threads = 0;
int completed_threads = 0;

std::random_device rd;
std::mt19937 gen(rd());

// Define constants
constexpr std::array<double, 6> p_values = {0.318, 0.3185, 0.319, 0.343, 0.3435, 0.344};
constexpr int L = 1024; // 2^10 = 1024
constexpr int STEPS_PER_LATTICEPOINT = 2000;
constexpr int RECORDING_INTERVAL = 10;
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
                int x = i + (k % 2) * 2 - 1;
                int y = j + (k / 2) * 2 - 1;

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

void run_csd(std::ofstream &file, double p)
{
    std::vector<bool> lattice = initLattice();

    for (int step = 1; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        updateLattice(lattice, p);

        if (step % (RECORDING_INTERVAL) == 0)
        {
            if (step >= RECORDING_STEP)
            {
                std::pair<std::vector<int>, std::vector<int>> cluster_sizes = get_cluster_sizes(lattice);

                // Write the data for this step
                file << step << "\t";
                for (size_t i = 0; i < cluster_sizes.first.size(); ++i)
                {
                    file << cluster_sizes.first[i];
                    if (i != cluster_sizes.first.size() - 1)
                    {
                        file << ",";
                    }
                }
                file << "\t";
                for (size_t i = 0; i < cluster_sizes.second.size(); ++i)
                {
                    file << cluster_sizes.second[i];
                    if (i != cluster_sizes.second.size() - 1)
                    {
                        file << ",";
                    }
                }
                file << "\n";
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
            filename << exeDir << "/outputs/CSD2D/criticalPointsCPU/p_" << p  << "_L_" << L << ".tsv";

            std::ofstream file(filename.str());

            file << "step\tfilled_cluster_size\tempty_cluster_size\n";

            run_csd(file, p);

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