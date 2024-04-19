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
constexpr int L = 10;               // 2^10 = 1024
constexpr bool TARGET_SITE = true;  // true = filled sites, false = empty sites
constexpr int MAX_BOX_SIZE = L / 4; // maximum box size for box counting method

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, 1);

std::vector<bool> initLattice(int L)
{
    std::vector<bool> soil_lattice(L * L);
    for (int i = 0; i < L * L; ++i)
    {
        soil_lattice[i] = dis(gen);
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

// std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> get_cluster_data(const std::vector<bool> &lattice, int L)
// {
//     UnionFind uf(L * L);
//     for (int i = 0; i < L; ++i)
//     {
//         for (int j = 0; j < L; ++j)
//         {
//             if (lattice[i * L + j] == TARGET_SITE)
//             {
//                 if (lattice[((i - 1 + L) % L) * L + j] == TARGET_SITE)
//                 {
//                     uf.union_set(i * L + j, ((i - 1 + L) % L) * L + j);
//                 }
//                 if (lattice[i * L + ((j - 1 + L) % L)] == TARGET_SITE)
//                 {
//                     uf.union_set(i * L + j, i * L + ((j - 1 + L) % L));
//                 }
//                 if (lattice[((i + 1) % L) * L + j] == TARGET_SITE)
//                 {
//                     uf.union_set(i * L + j, ((i + 1) % L) * L + j);
//                 }
//                 if (lattice[i * L + ((j + 1) % L)] == TARGET_SITE)
//                 {
//                     uf.union_set(i * L + j, i * L + ((j + 1) % L));
//                 }
//             }
//         }
//     }

//     std::unordered_map<int, std::pair<int, std::pair<int, int>>> cluster_sizes;
//     std::unordered_map<int, std::vector<std::pair<int, int>>> cluster_sites;
//     std::vector<int> labels;
//     for (int i = 0; i < L; ++i)
//     {
//         for (int j = 0; j < L; ++j)
//         {
//             if (lattice[i * L + j] == TARGET_SITE)
//             {
//                 int root = uf.find(i * L + j);
//                 labels.push_back(root);
//                 ++cluster_sizes[root].first;
//                 cluster_sites[root].push_back({i, j});
//             }
//         }
//     }

//     std::vector<int> sizes;
//     std::vector<int> max_dims;
//     for (const auto &pair : cluster_sizes)
//     {
//         sizes.push_back(pair.second.first);

//         std::vector<int> i_coords, j_coords;
//         const auto &sites = cluster_sites[pair.first];
//         for (const auto &site : sites)
//         {
//             i_coords.push_back(site.first);
//             j_coords.push_back(site.second);
//         }
//         std::sort(i_coords.begin(), i_coords.end());
//         std::sort(j_coords.begin(), j_coords.end());

//         int dx = 0;
//         for (size_t i = 0; i < i_coords.size(); ++i)
//         {
//             for (size_t j = i + 1; j < i_coords.size(); ++j)
//             {
//                 int dist = std::abs(i_coords[j] - i_coords[i]);
//                 dist = std::min(dist, L - dist); // take into account periodic boundary conditions
//                 dx = std::max(dx, dist);
//             }
//         }

//         int dy = 0;
//         for (size_t i = 0; i < j_coords.size(); ++i)
//         {
//             for (size_t j = i + 1; j < j_coords.size(); ++j)
//             {
//                 int dist = std::abs(j_coords[j] - j_coords[i]);
//                 dist = std::min(dist, L - dist); // take into account periodic boundary conditions
//                 dy = std::max(dy, dist);
//             }
//         }

//         int max_dim = std::max(dx, dy) + 1;
//         max_dims.push_back(max_dim);
//     }

//     return {sizes, max_dims, labels};
// }

// std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> get_cluster_data(const std::vector<bool> &lattice, int L)
// {
//     UnionFind uf(L * L);
//     for (int i = 0; i < L; ++i)
//     {
//         for (int j = 0; j < L; ++j)
//         {
//             if (lattice[i * L + j] == TARGET_SITE)
//             {
//                 uf.union_set(i * L + j, ((i - 1 + L) % L) * L + j);
//                 uf.union_set(i * L + j, i * L + ((j - 1 + L) % L));
//                 uf.union_set(i * L + j, ((i + 1) % L) * L + j);
//                 uf.union_set(i * L + j, i * L + ((j + 1) % L));
//             }
//         }
//     }

//     std::unordered_map<int, std::pair<int, std::pair<int, int>>> cluster_sizes;
//     std::unordered_map<int, std::vector<std::pair<int, int>>> cluster_sites;
//     std::vector<int> labels;
//     for (int i = 0; i < L; ++i)
//     {
//         for (int j = 0; j < L; ++j)
//         {
//             if (lattice[i * L + j] == TARGET_SITE)
//             {
//                 int root = uf.find(i * L + j);
//                 labels.push_back(root);
//                 ++cluster_sizes[root].first;
//                 cluster_sites[root].push_back({i, j});
//             }
//         }
//     }

//    std::vector<int> sizes;
//     std::vector<int> max_dims;
//     for (const auto &pair : cluster_sizes)
//     {
//         sizes.push_back(pair.second.first);

//         std::vector<int> i_coords, j_coords;
//         const auto &sites = cluster_sites[pair.first];
//         for (const auto &site : sites)
//         {
//             i_coords.push_back(site.first);
//             j_coords.push_back(site.second);
//         }
//         std::sort(i_coords.begin(), i_coords.end());
//         std::sort(j_coords.begin(), j_coords.end());

//         int dx = std::min(i_coords.back() - i_coords.front(), L - (i_coords.back() - i_coords.front()));
//         int dy = std::min(j_coords.back() - j_coords.front(), L - (j_coords.back() - j_coords.front()));

//         int max_dim = std::max(dx, dy) + 1; // +1 because x|x|x|x|x is 5 points but 4 intervals
//         max_dims.push_back(max_dim);
//     }

//     return {sizes, max_dims, labels};
// }

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

int main(int argc, char *argv[])
{
    std::vector<bool> lattice = initLattice(L);

    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> cluster_data = get_cluster_data(lattice, L);

    // output max dim, cluster size in console

    std::vector<int> sizes = std::get<0>(cluster_data);
    std::vector<int> max_dims = std::get<1>(cluster_data);

    std::cout << "Cluster size, Max dim" << std::endl;
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        std::cout << sizes[i] << ", " << max_dims[i] << std::endl;
    }

    for (int i = 0; i < L * L; ++i)
    {
        std::cout << lattice[i] << ",";
        if ((i + 1) % L == 0)
        {
            std::cout << "],\n[";
        }
    }

    return 0;
}