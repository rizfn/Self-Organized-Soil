#include <random>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>

// #pragma GCC optimize("Ofast","inline","fast-math","unroll-loops","no-stack-protector")
#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::random_device rd;
std::mt19937 gen(rd());

// Define a struct for coordinates
struct Coordinate
{
    int x;
    int y;
};

// Define constants
constexpr int L = 4096; // side length of the square lattice
constexpr double RHO_1 = 0.125;
constexpr double RHO_2 = 1;
constexpr double SIGMA = 1;
constexpr double THETA = 0.134;
constexpr int STEPS_PER_LATTICEPOINT = 6000;
constexpr int RECORDING_STEP = 3 * STEPS_PER_LATTICEPOINT / 4;
constexpr int RECORDING_INTERVAL = 20;

constexpr int EMPTY = 0;
constexpr int NUTRIENT = 1;
constexpr int SOIL = 2;
constexpr int WORM = 3;
constexpr int PARASITE = 4;

// Define distributions
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, 4);
std::uniform_int_distribution<> dis_l(0, L - 1);
std::uniform_real_distribution<> dis_real(0.0, 1.0);

Coordinate get_random_neighbour(Coordinate c, int L)
{
    // choose a random coordinate to change
    int coord_changing = dis(gen);
    // choose a random direction to change the coordinate
    int change = 2 * dis(gen) - 1;
    // change the coordinate
    c.x = (coord_changing == 0) ? (c.x + change + L) % L : c.x;
    c.y = (coord_changing == 1) ? (c.y + change + L) % L : c.y;

    return c;
}

std::vector<std::vector<int>> init_lattice(int L)
{
    std::vector<std::vector<int>> soil_lattice(L, std::vector<int>(L));
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            soil_lattice[i][j] = dis_site(gen);
        }
    }
    return soil_lattice;
}

void update(std::vector<std::vector<int>> &soil_lattice, int L, double sigma, double theta, double rho1, double rho2)
{
    // select a random site
    Coordinate site = {dis_l(gen), dis_l(gen)};

    if (soil_lattice[site.x][site.y] == EMPTY || soil_lattice[site.x][site.y] == NUTRIENT)
    { // empty or nutrient
        // choose a random neighbour
        Coordinate nbr = get_random_neighbour(site, L);
        if (soil_lattice[nbr.x][nbr.y] == SOIL)
        { // if neighbour is soil
            // fill with soil-filling rate
            if (dis_real(gen) < sigma)
            {
                soil_lattice[site.x][site.y] = SOIL;
            }
        }
    }
    else if (soil_lattice[site.x][site.y] == WORM || soil_lattice[site.x][site.y] == PARASITE)
    { // worm or parasite
        // check for death
        if (dis_real(gen) < theta)
        {
            soil_lattice[site.x][site.y] = EMPTY;
        }
        else
        {
            // move into a neighbour
            Coordinate new_site = get_random_neighbour(site, L);
            // check the value of the new site
            int new_site_value = soil_lattice[new_site.x][new_site.y];
            // determine species type
            bool is_parasite = (soil_lattice[site.x][site.y] == PARASITE);
            double current_rho = is_parasite ? rho2 : rho1;

            // move the worm or parasite
            soil_lattice[new_site.x][new_site.y] = soil_lattice[site.x][site.y];
            soil_lattice[site.x][site.y] = EMPTY;

            // check if the new site is nutrient
            if (new_site_value == NUTRIENT)
            {
                // reproduce behind you
                if (dis_real(gen) < current_rho)
                {
                    soil_lattice[site.x][site.y] = soil_lattice[new_site.x][new_site.y];
                }
            }
            // check if the new site is soil
            else if (new_site_value == SOIL)
            {
                // leave nutrient behind
                if (!is_parasite)
                {
                    soil_lattice[site.x][site.y] = NUTRIENT;
                }
            }
            // check if the new site is a worm or parasite
            else if (new_site_value == WORM || new_site_value == PARASITE)
            {
                // keep both with worms/parasites (undo the vacant space in original site)
                soil_lattice[site.x][site.y] = new_site_value;
            }
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

std::vector<int> get_soil_cluster_sizes(const std::vector<std::vector<int>> &soil_lattice)
{
    UnionFind uf(L * L);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if (soil_lattice[i][j] == SOIL)
            {
                if (soil_lattice[(i - 1 + L) % L][j] == SOIL)
                    uf.union_set(i * L + j, ((i - 1 + L) % L) * L + j);
                if (soil_lattice[i][(j - 1 + L) % L] == SOIL)
                    uf.union_set(i * L + j, i * L + ((j - 1 + L) % L));
                if (soil_lattice[(i + 1) % L][j] == SOIL)
                    uf.union_set(i * L + j, ((i + 1) % L) * L + j);
                if (soil_lattice[i][(j + 1) % L] == SOIL)
                    uf.union_set(i * L + j, i * L + ((j + 1) % L));
            }
        }
    }

    std::unordered_map<int, int> cluster_sizes;
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if (soil_lattice[i][j] == SOIL)
            {
                int root = uf.find(i * L + j);
                ++cluster_sizes[root];
            }
        }
    }

    std::vector<int> sizes;
    for (const auto &pair : cluster_sizes)
        sizes.push_back(pair.second);
    return sizes;
}

void run_csd(std::ofstream &file, double sigma, double theta, double rho1, double rho2)
{
    std::vector<std::vector<int>> soil_lattice = init_lattice(L);

    for (int step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            update(soil_lattice, L, sigma, theta, rho1, rho2);
        }

        if ((step >= RECORDING_STEP) && (step % RECORDING_INTERVAL == 0))
        {
            std::vector<int> cluster_sizes = get_soil_cluster_sizes(soil_lattice);

            file << step;
            for (size_t i = 0; i < cluster_sizes.size(); ++i)
            {
                file << (i == 0 ? "\t" : ",") << cluster_sizes[i];
            }
            file << "\n";
        }

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / STEPS_PER_LATTICEPOINT * 100 << "%\r" << std::flush;
    }
}

int main(int argc, char *argv[])
{
    double sigma = SIGMA;
    double theta = THETA;
    double rho1 = RHO_1;
    double rho2 = RHO_2;
    if (argc > 1)
        sigma = std::stod(argv[1]);
    if (argc > 2)
        theta = std::stod(argv[2]);
    if (argc > 3)
        rho1 = std::stod(argv[3]);
    if (argc > 4)
        rho2 = std::stod(argv[4]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "\\outputs\\csd2D_high_s\\sigma_" << sigma << "_theta_" << theta << ".tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "step\tcsd_soil\n";

    run_csd(file, sigma, theta, rho1, rho2);

    file.close();

    return 0;
}