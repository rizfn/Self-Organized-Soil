#include <random>
#include <vector>
#include <unordered_map>
#include <algorithm>
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
constexpr double THETA = 0.377;
constexpr int L = 4096;             // 2^10 = 1024
constexpr bool TARGET_SITE = true;  // true = filled sites, false = empty sites
constexpr int MAX_BOX_SIZE = L / 4; // maximum box size for box counting method
constexpr int STEPS_PER_LATTICEPOINT = 3000;
constexpr int RECORDING_INTERVAL = 20;
constexpr int RECORDING_STEP = 2000;

std::random_device rd;
std::mt19937 gen(rd());
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

void updateLattice(std::vector<bool> &lattice)
{
    // Choose a random site
    int site_index = dis_site(gen);
    Coordinate site = {site_index % L, site_index / L};

    if (lattice[site_index]) // if the site is occupied
    {
        // make it empty with rate THETA
        if (dis_prob(gen) < THETA)
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

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> get_cluster_data(const std::vector<bool> &lattice, int L)
{
    UnionFind uf(L * L);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if (lattice[i * L + j] == TARGET_SITE)
            {
                if (lattice[((i - 1 + L) % L) * L + j] == TARGET_SITE)
                {
                    uf.union_set(i * L + j, ((i - 1 + L) % L) * L + j);
                }
                if (lattice[i * L + ((j - 1 + L) % L)] == TARGET_SITE)
                {
                    uf.union_set(i * L + j, i * L + ((j - 1 + L) % L));
                }
                if (lattice[((i + 1) % L) * L + j] == TARGET_SITE)
                {
                    uf.union_set(i * L + j, ((i + 1) % L) * L + j);
                }
                if (lattice[i * L + ((j + 1) % L)] == TARGET_SITE)
                {
                    uf.union_set(i * L + j, i * L + ((j + 1) % L));
                }
            }
        }
    }

    std::unordered_map<int, std::pair<int, std::pair<int, int>>> cluster_sizes;
    std::unordered_map<int, std::vector<std::pair<int, int>>> cluster_sites;
    std::vector<int> labels;
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if (lattice[i * L + j] == TARGET_SITE)
            {
                int root = uf.find(i * L + j);
                labels.push_back(root);
                ++cluster_sizes[root].first;
                cluster_sites[root].push_back({i, j});
            }
        }
    }

    std::vector<int> sizes;
    std::vector<int> max_dims;
    for (const auto &pair : cluster_sizes)
    {
        sizes.push_back(pair.second.first);

        std::vector<int> i_coords, j_coords;
        const auto &sites = cluster_sites[pair.first];
        for (const auto &site : sites)
        {
            i_coords.push_back(site.first);
            j_coords.push_back(site.second);
        }
        std::sort(i_coords.begin(), i_coords.end());
        std::sort(j_coords.begin(), j_coords.end());

        int dx = i_coords.back() - i_coords.front();
        if (i_coords.front() == 0 && i_coords.back() == L - 1)
        {
            auto it = std::adjacent_find(i_coords.begin(), i_coords.end(), [](int a, int b)
                                         { return b - a > 1; });
            if (it != i_coords.end())
            {
                dx = L - (*std::next(it) - *it);
            }
            else
            {
                dx = L - 1;
            }
        }

        int dy = j_coords.back() - j_coords.front();
        if (j_coords.front() == 0 && j_coords.back() == L - 1)
        {
            auto it = std::adjacent_find(j_coords.begin(), j_coords.end(), [](int a, int b)
                                         { return b - a > 1; });
            if (it != j_coords.end())
            {
                dy = L - (*std::next(it) - *it);
            }
            else
            {
                dy = L - 1;
            }
        }
        int max_dim = std::max(dx, dy) + 1;
        max_dims.push_back(max_dim);

        // print error if cluster size not in [max_dim, max_dim^2]
        if (sizes.back() < max_dim || sizes.back() > max_dim * max_dim)
        {
            std::cerr << "Cluster size " << sizes.back() << " not in [" << max_dim << ", " << max_dim * max_dim << "]\n";
            std::cerr << "dx: " << dx << ", dy: " << dy << "\n";
            std::cerr << "i_coords: ";
            for (const auto &i : i_coords)
            {
                std::cerr << i << " ";
            }
            std::cerr << "\n";

            std::cerr << "j_coords: ";
            for (const auto &j : j_coords)
            {
                std::cerr << j << " ";
            }
            std::cerr << "\n";
        }
    }

    return {sizes, max_dims, labels};
}


std::pair<std::vector<int>, std::vector<int>> calculate_fractal_dimension(const std::vector<bool> &lattice)
{
    std::vector<int> box_sizes;
    std::vector<int> counts;
    for (int s = 1; s <= MAX_BOX_SIZE; s *= 2)
    {
        int count = 0;
        for (int i = 0; i < L; i += s)
        {
            for (int j = 0; j < L; j += s)
            {
                for (int x = i; x < std::min(i + s, L); ++x)
                {
                    for (int y = j; y < std::min(j + s, L); ++y)
                    {
                        if (lattice[x * L + y] == TARGET_SITE)
                        {
                            ++count;
                            goto next_box;
                        }
                    }
                }
            next_box:;
            }
        }
        box_sizes.push_back(s);
        counts.push_back(count);
    }
    return {box_sizes, counts};
}

void run_huber(std::ofstream &file)
{
    std::vector<bool> lattice = initLattice(L);

    for (int step = 1; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            updateLattice(lattice);
        }
        if (step % (RECORDING_INTERVAL) == 0 && step >= RECORDING_STEP)
        {
            // Get cluster data
            std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> cluster_data = get_cluster_data(lattice, L);

            // Calculate fractal dimensions
            std::pair<std::vector<int>, std::vector<int>> fractal_dims = calculate_fractal_dimension(lattice);

            // Create a copy of the lattice where each cluster is represented by a single point
            std::vector<bool> point_lattice(L * L, false);
            for (const auto &label : std::get<2>(cluster_data))
            {
                point_lattice[label] = true;
            }

            // Calculate the fractal dimension of the point lattice
            std::pair<std::vector<int>, std::vector<int>> point_fractal_dims = calculate_fractal_dimension(point_lattice);

            // Write the data for this step
            file << step << "\t";
            const auto &sizes = std::get<0>(cluster_data);
            for (size_t i = 0; i < sizes.size(); ++i)
            {
                file << sizes[i];
                if (i != sizes.size() - 1)
                {
                    file << ",";
                }
            }
            file << "\t";
            const auto &dims = std::get<1>(cluster_data);
            for (size_t i = 0; i < dims.size(); ++i)
            {
                file << dims[i];
                if (i != dims.size() - 1)
                {
                    file << ",";
                }
            }
            file << "\t";
            for (size_t i = 0; i < fractal_dims.first.size(); ++i)
            {
                file << fractal_dims.first[i];
                if (i != fractal_dims.first.size() - 1)
                {
                    file << ",";
                }
            }
            file << "\t";
            for (size_t i = 0; i < fractal_dims.second.size(); ++i)
            {
                file << fractal_dims.second[i];
                if (i != fractal_dims.second.size() - 1)
                {
                    file << ",";
                }
            }
            file << "\t";
            for (size_t i = 0; i < point_fractal_dims.second.size(); ++i)
            {
                file << point_fractal_dims.second[i];
                if (i != point_fractal_dims.second.size() - 1)
                {
                    file << ",";
                }
            }
            file << "\n";
        }
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / STEPS_PER_LATTICEPOINT * 100 << "%\r" << std::flush;
    }
}

int main(int argc, char *argv[])
{
    std::vector<bool> lattice = initLattice(L);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filename;
    filename << exeDir << "/outputs/huber/sigma_" << SIGMA << "_theta_" << THETA << "_target_" << TARGET_SITE << ".tsv";

    std::ofstream file(filename.str());

    file << "step\tcluster_sizes\tcluster_lineardim\tbox_sizes\tfractal_dim\tpoint_fractal_dim\n";

    run_huber(file);

    file.close();

    return 0;
}