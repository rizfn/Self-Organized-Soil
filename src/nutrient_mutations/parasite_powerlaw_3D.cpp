#include <bits/stdc++.h>
#include <windows.h>

// #pragma GCC optimize("Ofast","inline","fast-math","unroll-loops","no-stack-protector")
#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::random_device rd;
std::mt19937 gen(rd());

// Define a struct for coordinates
struct Coordinate
{
    int x;
    int y;
    int z;
};

// Define constants
constexpr int STEPS_PER_LATTICEPOINT = 1000;
constexpr double SIGMA = 0.5;
constexpr double THETA = 0.025;
constexpr double RHO1 = 0.5;
constexpr double MU1 = 0.5;
constexpr double RHO2 = 1;
constexpr double MU2 = 0;
constexpr int L = 100; // side length of the cubic lattice

constexpr int EMPTY = 0;
constexpr int NUTRIENT = 1;
constexpr int SOIL = 2;
constexpr int GREEN_WORM = 3;
constexpr int BLUE_WORM = 4;

// Define distributions
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_coord(0, 2);
std::uniform_int_distribution<> dis_site(0, 4);
std::uniform_int_distribution<> dis_l(0, L - 1);
std::uniform_real_distribution<> dis_real(0.0, 1.0);

Coordinate get_random_neighbour(Coordinate c, int L)
{
    // choose a random coordinate to change
    int coord_changing = dis_coord(gen);
    // choose a random direction to change the coordinate
    int change = 2 * dis(gen) - 1;
    // change the coordinate
    c.x = (coord_changing == 0) ? (c.x + change + L) % L : c.x;
    c.y = (coord_changing == 1) ? (c.y + change + L) % L : c.y;
    c.z = (coord_changing == 2) ? (c.z + change + L) % L : c.z; // Added z coordinate change

    return c;
}

std::vector<std::vector<std::vector<int>>> init_lattice(int L)
{
    std::vector<std::vector<std::vector<int>>> soil_lattice(L, std::vector<std::vector<int>>(L, std::vector<int>(L))); // Changed to 3D vector
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int k = 0; k < L; ++k) // Added z coordinate loop
            {
                soil_lattice[i][j][k] = dis_site(gen);
            }
        }
    }
    return soil_lattice;
}

void update(std::vector<std::vector<std::vector<int>>> &soil_lattice, int L, double sigma, double theta, double rho1, double rho2, double mu1, double mu2)
{
    // select a random site
    Coordinate site = {dis_l(gen), dis_l(gen), dis_l(gen)};

    if (soil_lattice[site.x][site.y][site.z] == EMPTY || soil_lattice[site.x][site.y][site.z] == NUTRIENT)
    { // empty or nutrient
        // choose a random neighbour
        Coordinate nbr = get_random_neighbour(site, L);
        if (soil_lattice[nbr.x][nbr.y][nbr.z] == SOIL)
        { // if neighbour is soil
            // fill with soil-filling rate
            if (dis_real(gen) < sigma)
            {
                soil_lattice[site.x][site.y][site.z] = SOIL;
            }
        }
    }
    else if (soil_lattice[site.x][site.y][site.z] == GREEN_WORM)
    { // green worm
        // check for death
        if (dis_real(gen) < theta)
        {
            soil_lattice[site.x][site.y][site.z] = EMPTY;
        }
        else
        {
            // move into a neighbour
            Coordinate new_site = get_random_neighbour(site, L);
            // check the value of the new site
            int new_site_value = soil_lattice[new_site.x][new_site.y][new_site.z];
            // move the worm
            soil_lattice[new_site.x][new_site.y][new_site.z] = GREEN_WORM;
            soil_lattice[site.x][site.y][site.z] = EMPTY;
            // check if the new site is nutrient
            if (new_site_value == NUTRIENT)
            {
                // reproduce behind you
                if (dis_real(gen) < rho1)
                {
                    soil_lattice[site.x][site.y][site.z] = GREEN_WORM;
                }
            }
            // check if the new site is soil
            else if (new_site_value == SOIL)
            {
                // leave nutrient behind
                if (dis_real(gen) < mu1)
                {
                    soil_lattice[site.x][site.y][site.z] = NUTRIENT;
                }
            }
            // check if the new site is a worm
            else if (new_site_value == GREEN_WORM || new_site_value == BLUE_WORM)
            {
                // keep both with worms (undo the vacant space in original site)
                soil_lattice[site.x][site.y][site.z] = new_site_value;
            }
        }
    }
    else if (soil_lattice[site.x][site.y][site.z] == BLUE_WORM)
    { // blue worm
        // check for death
        if (dis_real(gen) < theta)
        {
            soil_lattice[site.x][site.y][site.z] = EMPTY;
        }
        else
        {
            // move into a neighbour
            Coordinate new_site = get_random_neighbour(site, L);
            // check the value of the new site
            int new_site_value = soil_lattice[new_site.x][new_site.y][new_site.z];
            // move the worm
            soil_lattice[new_site.x][new_site.y][new_site.z] = BLUE_WORM;
            soil_lattice[site.x][site.y][site.z] = EMPTY;
            // check if the new site is nutrient
            if (new_site_value == NUTRIENT)
            {
                // reproduce behind you
                if (dis_real(gen) < rho2)
                {
                    soil_lattice[site.x][site.y][site.z] = BLUE_WORM;
                }
            }
            // check if the new site is soil
            else if (new_site_value == SOIL)
            {
                // leave nutrient behind
                if (dis_real(gen) < mu2)
                {
                    soil_lattice[site.x][site.y][site.z] = NUTRIENT;
                }
            }
            // check if the new site is a worm
            else if (new_site_value == GREEN_WORM || new_site_value == BLUE_WORM)
            {
                // keep both with worms (undo the vacant space in original site)
                soil_lattice[site.x][site.y][site.z] = new_site_value;
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

std::vector<int> get_cluster_sizes(const std::vector<std::vector<std::vector<int>>> &soil_lattice, int L)
{
    UnionFind uf(L * L * L); // Changed to L * L * L
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int k = 0; k < L; ++k) // Added z coordinate loop
            {
                if (soil_lattice[i][j][k] == 2)
                {
                    if (i > 0 && soil_lattice[i - 1][j][k] == 2)
                        uf.union_set(i * L * L + j * L + k, (i - 1) * L * L + j * L + k); // Changed to L * L
                    if (j > 0 && soil_lattice[i][j - 1][k] == 2)
                        uf.union_set(i * L * L + j * L + k, i * L * L + (j - 1) * L + k); // Changed to L * L
                    if (k > 0 && soil_lattice[i][j][k - 1] == 2)
                        uf.union_set(i * L * L + j * L + k, i * L * L + j * L + k - 1); // Added z coordinate union
                }
            }
        }
    }

    std::unordered_map<int, int> cluster_sizes;
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int k = 0; k < L; ++k) // Added z coordinate loop
            {
                if (soil_lattice[i][j][k] == 2)
                {
                    int root = uf.find(i * L * L + j * L + k); // Changed to L * L
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

void run_csd(int n_steps, int L, double sigma, double theta, double rho1, double rho2, double mu1, double mu2, std::ofstream &file)
{
    std::vector<std::vector<std::vector<int>>> soil_lattice = init_lattice(L); // Changed to 3D vector

    for (int step = 0; step <= n_steps; ++step)
    {
        update(soil_lattice, L, sigma, theta, rho1, rho2, mu1, mu2); // Updated to handle 3D lattice
        if (step % (L * L * L) == 0)                                 // Changed to L * L * L
        {
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / n_steps * 100 << "%\r" << std::flush;
            std::vector<int> cluster_sizes = get_cluster_sizes(soil_lattice, L); // Updated to handle 3D lattice

            // Write the data for this step
            file << step * L * L * L; // Changed to L * L * L
            for (const auto &size : cluster_sizes)
            {
                file << "," << size;
            }
            file << "\n";
        }
    }
}

int main(int argc, char *argv[])
{
    // initialize the parameters
    int n_steps = STEPS_PER_LATTICEPOINT * L * L * L; // 3D

    // Check if command line arguments are provided
    double sigma = SIGMA;
    double theta = THETA;
    if (argc > 1)
        sigma = std::stod(argv[1]);
    if (argc > 2)
        theta = std::stod(argv[2]);

    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filename;
    filename << exeDir << "\\outputs\\parasite_CSD_3D\\sigma_" << sigma << "_theta_" << theta << ".csv";
    std::string filePath = filename.str();

    std::ofstream file;
    file.open(filePath);

    // Write the header
    file << "step,cluster_sizes\n";

    // Run the time series
    run_csd(n_steps, L, sigma, theta, RHO1, RHO2, MU1, MU2, file);

    file.close();

    return 0;
}