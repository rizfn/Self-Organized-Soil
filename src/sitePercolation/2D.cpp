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

std::random_device rd;
std::mt19937 gen(rd());

// Define constants
constexpr std::array<double, 4> p_values = {0.575, 0.592746, 0.6, 0.62};
constexpr int L = 4096; // 2^10 = 1024
constexpr int N_TRIALS = 8;

thread_local std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<bool> initLattice(double p)
{
    std::vector<bool> soil_lattice(L * L, false); // Initialize all cells to false
    for (int i = 0; i < L * L; ++i)
    {
        soil_lattice[i] = dis_prob(gen) < p;
    }
    return soil_lattice;
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
    UnionFind uf_filled(L * L);
    UnionFind uf_empty(L * L);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if (lattice[i * L + j])
            {
                if (lattice[((i - 1 + L) % L) * L + j])
                    uf_filled.union_set(i * L + j, ((i - 1 + L) % L) * L + j);
                if (lattice[i * L + ((j - 1 + L) % L)])
                    uf_filled.union_set(i * L + j, i * L + ((j - 1 + L) % L));
                if (lattice[((i + 1) % L) * L + j])
                    uf_filled.union_set(i * L + j, ((i + 1) % L) * L + j);
                if (lattice[i * L + ((j + 1) % L)])
                    uf_filled.union_set(i * L + j, i * L + ((j + 1) % L));
            }
            else
            {
                if (!lattice[((i - 1 + L) % L) * L + j])
                    uf_empty.union_set(i * L + j, ((i - 1 + L) % L) * L + j);
                if (!lattice[i * L + ((j - 1 + L) % L)])
                    uf_empty.union_set(i * L + j, i * L + ((j - 1 + L) % L));
                if (!lattice[((i + 1) % L) * L + j])
                    uf_empty.union_set(i * L + j, ((i + 1) % L) * L + j);
                if (!lattice[i * L + ((j + 1) % L)])
                    uf_empty.union_set(i * L + j, i * L + ((j + 1) % L));
            }
        }
    }

    std::unordered_map<int, int> cluster_sizes_filled;
    std::unordered_map<int, int> cluster_sizes_empty;
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if (lattice[i * L + j])
            {
                int root = uf_filled.find(i * L + j);
                ++cluster_sizes_filled[root];
            }
            else
            {
                int root = uf_empty.find(i * L + j);
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

    for (int i = 0; i < N_TRIALS; ++i)
    {
        std::vector<bool> lattice = initLattice(p);
        std::pair<std::vector<int>, std::vector<int>> cluster_sizes = get_cluster_sizes(lattice);
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

int main(int argc, char *argv[])
{
    std::vector<std::thread> threads;

    for (double p : p_values)
    {
        threads.push_back(std::thread([p, &argv]() { // Capture p by value
            std::string exePath = argv[0];
            std::string exeDir = std::filesystem::path(exePath).parent_path().string();

            std::ostringstream filename;
            // filename << exeDir << "/outputs/CSD2D/p_" << p << "_L_" << L << ".tsv";
            filename << exeDir << "/outputs/CSD2D_ntrials/p_" << p << "_L_" << L << "_n_" << N_TRIALS << ".tsv";

            std::ofstream file(filename.str());

            file << "filled_cluster_size\tempty_cluster_size\n";

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