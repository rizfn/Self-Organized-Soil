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

thread_local std::random_device rd;
thread_local std::mt19937 gen(rd());

// Define constants
constexpr double SIGMA = 1;
constexpr std::array<double, 6> theta_values = {0.26, 0.27, 0.28, 0.51, 0.52, 0.53};
constexpr int L = 1024; // 2^10 = 1024
constexpr long long STEPS_PER_LATTICEPOINT = 2000;
constexpr long long N_STEPS = STEPS_PER_LATTICEPOINT * L * L;
constexpr int RECORDING_INTERVAL = 20;
constexpr long long RECORDING_STEP = 1000 * L * L;

std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, L *L - 1);
std::uniform_int_distribution<> dis_dir(0, 3);
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<bool> initLattice(int L)
{

    std::vector<bool> soil_lattice(L * L);
    for (int i = 0; i < L * L; ++i)
    {
        soil_lattice[i] = dis(gen);
    }
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

std::pair<std::vector<int>, std::vector<int>> get_cluster_sizes(const std::vector<bool>& lattice, int L) {
    UnionFind uf_filled(L * L);
    UnionFind uf_empty(L * L);
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            if (lattice[i * L + j]) {
                uf_filled.union_set(i * L + j, ((i - 1 + L) % L) * L + j);
                uf_filled.union_set(i * L + j, i * L + ((j - 1 + L) % L));
                uf_filled.union_set(i * L + j, ((i + 1) % L) * L + j);
                uf_filled.union_set(i * L + j, i * L + ((j + 1) % L));
            } else {
                uf_empty.union_set(i * L + j, ((i - 1 + L) % L) * L + j);
                uf_empty.union_set(i * L + j, i * L + ((j - 1 + L) % L));
                uf_empty.union_set(i * L + j, ((i + 1) % L) * L + j);
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

void run_csd(std::ofstream &file, double theta)
{
    std::vector<bool> lattice = initLattice(L);

    for (int step = 1; step <= N_STEPS; ++step)
    {
        updateLattice(lattice, theta);
        if (step % (RECORDING_INTERVAL * L * L) == 0)
        {
            if (step >= RECORDING_STEP)
            {
                std::pair<std::vector<int>, std::vector<int>> cluster_sizes = get_cluster_sizes(lattice, L);

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

    for (double theta : theta_values)
    {
        threads.push_back(std::thread([theta, &argv]() { // Capture theta by value
            std::vector<bool> lattice = initLattice(L);

            std::string exePath = argv[0];
            std::string exeDir = std::filesystem::path(exePath).parent_path().string();

            std::ostringstream filename;
            filename << exeDir << "/outputs/csd_criticalpoints/sigma_" << SIGMA << "_theta_" << theta << ".tsv";
            // filename << exeDir << "/outputs/csd_criticalpoints/small_system/sigma_" << SIGMA << "_theta_" << theta << ".tsv";

            std::ofstream file(filename.str());

            file << "step\tfilled_cluster_size\tempty_cluster_size\n";

            run_csd(file, theta);

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