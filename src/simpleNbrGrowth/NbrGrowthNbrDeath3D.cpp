#include <random>
#include <vector>
#include <unordered_map>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

// Define constants
constexpr double SIGMA = 1;
constexpr double THETA = 1;
constexpr int L = 100; // 2^10 = 1024
constexpr long long STEPS_PER_LATTICEPOINT = 2000;
constexpr long long N_STEPS = STEPS_PER_LATTICEPOINT * L * L * L;
constexpr int RECORDING_INTERVAL = 20;
constexpr long long RECORDING_STEP = (STEPS_PER_LATTICEPOINT / 2) * L * L * L;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, L *L *L - 1);
std::uniform_int_distribution<> dis_dir(0, 5);
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<bool> initLattice(int L)
{
    std::vector<bool> soil_lattice(L * L * L);
    for (int i = 0; i < L * L * L; ++i)
    {
        soil_lattice[i] = dis(gen);
    }
    return soil_lattice;
}

struct Coordinate
{
    int x;
    int y;
    int z;
};

Coordinate get_random_neighbour(Coordinate site, int L)
{
    int dir = dis_dir(gen);
    switch (dir)
    {
    case 0: // left
        return {(site.x - 1 + L) % L, site.y, site.z};
    case 1: // right
        return {(site.x + 1) % L, site.y, site.z};
    case 2: // above
        return {site.x, (site.y - 1 + L) % L, site.z};
    case 3: // below
        return {site.x, (site.y + 1) % L, site.z};
    case 4: // in front
        return {site.x, site.y, (site.z - 1 + L) % L};
    case 5: // behind
        return {site.x, site.y, (site.z + 1) % L};
    }
    return site; // should never reach here
}

void updateLattice(std::vector<bool> &lattice)
{
    // Choose a random site
    int site_index = dis_site(gen);
    Coordinate site = {site_index % L, (site_index / L) % L, site_index / (L * L)};

    if (lattice[site_index]) // if the site is occupied
    {
        // choose a random neighbour
        Coordinate nbr = get_random_neighbour(site, L);
        int nbr_index = nbr.z * L * L + nbr.y * L + nbr.x;

        // if the neighbour is empty, make it empty with rate THETA
        if (!lattice[nbr_index] && dis_prob(gen) < THETA)
        {
            lattice[site_index] = false;
        }
    }
    else // if the site is empty
    {
        // choose a random neighbour
        Coordinate nbr = get_random_neighbour(site, L);
        int nbr_index = nbr.z * L * L + nbr.y * L + nbr.x;

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

std::vector<int> get_cluster_sizes(const std::vector<bool> &lattice, int L)
{
    UnionFind uf(L * L * L);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int k = 0; k < L; ++k)
            {
                if (lattice[i * L * L + j * L + k])
                {
                    if (i > 0 && lattice[(i - 1) * L * L + j * L + k])
                        uf.union_set(i * L * L + j * L + k, (i - 1) * L * L + j * L + k);
                    if (j > 0 && lattice[i * L * L + (j - 1) * L + k])
                        uf.union_set(i * L * L + j * L + k, i * L * L + (j - 1) * L + k);
                    if (k > 0 && lattice[i * L * L + j * L + k - 1])
                        uf.union_set(i * L * L + j * L + k, i * L * L + j * L + k - 1);
                }
            }
        }
    }

    std::unordered_map<int, int> cluster_sizes;
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int k = 0; k < L; ++k)
            {
                if (lattice[i * L * L + j * L + k])
                {
                    int root = uf.find(i * L * L + j * L + k);
                    ++cluster_sizes[root];
                }
            }
        }
    }

    std::vector<int> sizes;
    for (const auto &pair : cluster_sizes)
        sizes.push_back(pair.second);
    return sizes;
}

void run_csd(std::ofstream &file)
{
    std::vector<bool> lattice = initLattice(L);

    for (long long step = 1; step <= N_STEPS; ++step)
    {
        updateLattice(lattice);
        if (step % (static_cast<long long>(RECORDING_INTERVAL) * L * L * L) == 0)
        {
            if (step >= RECORDING_STEP)
            {
                std::vector<int> cluster_sizes = get_cluster_sizes(lattice, L);

                // Write the data for this step
                file << step;
                for (const auto &size : cluster_sizes)
                {
                    file << "," << size;
                }
                file << "\n";
            }
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / N_STEPS * 100 << "%\r" << std::flush;
        }
    }
}

int main(int argc, char *argv[])
{
    std::vector<bool> lattice = initLattice(L);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filename;
    filename << exeDir << "/outputs/csdNbrDeath/3D_sigma_" << SIGMA << "_theta_" << THETA << ".csv";

    std::ofstream file(filename.str());

    file << "step,cluster_size\n";

    run_csd(file);

    file.close();

    return 0;
}