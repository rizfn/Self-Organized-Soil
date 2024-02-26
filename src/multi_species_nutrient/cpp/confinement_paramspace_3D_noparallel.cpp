#include <bits/stdc++.h>
#include <windows.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <condition_variable>

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
constexpr int STEPS_PER_LATTICEPOINT = 20000;
constexpr double SIGMA = 0.5;
constexpr double THETA = 0.0391;
constexpr double RHO = 1;
constexpr double MU = 1;
constexpr int L = 50; // side length of the cubic lattice
constexpr long long N_STEPS = static_cast<long long>(STEPS_PER_LATTICEPOINT) * L * L * L;

constexpr int EMPTY = 0;

// Define distributions
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_coord(0, 2);
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
    c.z = (coord_changing == 2) ? (c.z + change + L) % L : c.z;

    return c;
}

std::vector<std::vector<std::vector<int>>> init_lattice(int L, int N)
{
    std::uniform_int_distribution<> dis_site(0, 2 * N + 1);

    std::vector<std::vector<std::vector<int>>> soil_lattice(L, std::vector<std::vector<int>>(L, std::vector<int>(L))); // Changed to 3D vector
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int k = 0; k < L; ++k)
            {
                soil_lattice[i][j][k] = dis_site(gen);
            }
        }
    }
    return soil_lattice;
}

void update(std::vector<std::vector<std::vector<int>>> &soil_lattice, int L, double sigma, double theta, double rho, double mu, const std::vector<int> &NUTRIENTS, const std::vector<int> &WORMS, int SOIL, int EMPTY, int N)
{
    // select a random site
    Coordinate site = {dis_l(gen), dis_l(gen), dis_l(gen)};

    if (soil_lattice[site.x][site.y][site.z] == EMPTY || std::find(NUTRIENTS.begin(), NUTRIENTS.end(), soil_lattice[site.x][site.y][site.z]) != NUTRIENTS.end())
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
    else
    {
        for (int i = 0; i < N; ++i)
        {
            if (soil_lattice[site.x][site.y][site.z] == WORMS[i])
            { // worm of species i
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
                    soil_lattice[new_site.x][new_site.y][new_site.z] = WORMS[i];
                    soil_lattice[site.x][site.y][site.z] = EMPTY;
                    // check if the new site is a nutrient that this worm can consume
                    if (new_site_value == NUTRIENTS[(i + 1) % N]) // Modified condition to check for nutrient to the right (with wrapping)
                    {
                        // reproduce behind you
                        if (dis_real(gen) < rho)
                        {
                            soil_lattice[site.x][site.y][site.z] = WORMS[i];
                        }
                    }
                    // check if the new site is soil
                    else if (new_site_value == SOIL)
                    {
                        // leave nutrient behind
                        if (dis_real(gen) < mu)
                        {
                            soil_lattice[site.x][site.y][site.z] = NUTRIENTS[i];
                        }
                    }
                    // check if the new site is a worm
                    else if (std::find(WORMS.begin(), WORMS.end(), new_site_value) != WORMS.end())
                    {
                        // keep both with worms (undo the vacant space in original site)
                        soil_lattice[site.x][site.y][site.z] = new_site_value;
                    }
                }
            }
        }
    }
}

double count_soil_fraction(const std::vector<std::vector<std::vector<int>>> &soil_lattice, int SOIL)
{
    int total = 0;
    int soil = 0;
    for (const auto &matrix : soil_lattice)
    {
        for (const auto &row : matrix)
        {
            for (int cell : row)
            {
                if (cell == SOIL)
                {
                    ++soil;
                }
                ++total;
            }
        }
    }
    return static_cast<double>(soil) / total;
}

void run_and_write_results(int N, double theta, double sigma, const std::string &filePath)
{
    std::vector<int> NUTRIENTS(N);
    std::vector<int> WORMS(N);
    int SOIL = N + 1;
    for (int i = 0; i < N; ++i)
    {
        NUTRIENTS[i] = i + 1;
        WORMS[i] = i + N + 2;
    }
    std::vector<std::vector<std::vector<int>>> soil_lattice = init_lattice(L, N);
    for (long long step = 0; step <= N_STEPS; ++step)
    {
        update(soil_lattice, L, sigma, theta, RHO, MU, NUTRIENTS, WORMS, SOIL, EMPTY, N);
        if (step % (L * L * L) == 0)
        {
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / N_STEPS * 100 << "%\r" << std::flush;
        }
    }
    double soil_fraction = count_soil_fraction(soil_lattice, SOIL);

    std::ofstream file(filePath, std::ios_base::app);
    file << N << "," << theta << "," << sigma << "," << soil_fraction << "\n";

}

std::vector<double> linspace(double start, double end, int num)
{
    std::vector<double> linspaced;

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // Ensure that end is included

    return linspaced;
}

int main()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "\\..\\outputs\\confinement\\soilfracs_3D.csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "N,theta,sigma,soil_fraction\n";

    std::map<int, std::pair<double, double>> theta_ranges = {
        {2, {0.05, 0.06}},
        {3, {0.03, 0.05}},
        {4, {0.03, 0.04}},
        {5, {0.01, 0.02}},
        {6, {0.001, 0.01}},
        {7, {0.001, 0.01}},
        {8, {0.001, 0.01}}};

    for (int N = 2; N <= 8; ++N)
    {
        std::vector<double> theta_values = linspace(theta_ranges[N].first, theta_ranges[N].second, 5);
        for (double theta : theta_values)
        {
            run_and_write_results(N, theta, SIGMA, filePath);
            std::cout << "N = " << N << ", theta = " << theta << " done\n";
        }
    }

    return 0;
}