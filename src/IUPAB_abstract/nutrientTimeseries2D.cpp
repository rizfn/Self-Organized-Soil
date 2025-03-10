#include <bits/stdc++.h>
#include <windows.h>
#include <chrono>

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
};

// Define constants
constexpr int STEPS_PER_LATTICEPOINT = 4000;
constexpr double RHO1 = 1;
constexpr double MU1 = 1;
constexpr int L = 1024; // side length of the square lattice

constexpr int EMPTY = 0;
constexpr int NUTRIENT = 1;
constexpr int SOIL = 2;
constexpr int GREEN_WORM = 3;

constexpr double SIGMA = 0.5;
constexpr double THETA = 0.3;

// Define distributions
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, 3);
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

void update(std::vector<std::vector<int>> &soil_lattice, int L, double sigma, double theta, double rho1, double mu1)
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
    else if (soil_lattice[site.x][site.y] == GREEN_WORM)
    { // green worm
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
            // move the worm
            soil_lattice[new_site.x][new_site.y] = GREEN_WORM;
            soil_lattice[site.x][site.y] = EMPTY;
            // check if the new site is nutrient
            if (new_site_value == NUTRIENT)
            {
                // reproduce behind you
                if (dis_real(gen) < rho1)
                {
                    soil_lattice[site.x][site.y] = GREEN_WORM;
                }
            }
            // check if the new site is soil
            else if (new_site_value == SOIL)
            {
                // leave nutrient behind
                if (dis_real(gen) < mu1)
                {
                    soil_lattice[site.x][site.y] = NUTRIENT;
                }
            }
            // check if the new site is a worm
            else if (new_site_value == GREEN_WORM)
            {
                // keep both with worms (undo the vacant space in original site)
                soil_lattice[site.x][site.y] = new_site_value;
            }
        }
    }
}

void run_timeseries(int STEPS_PER_LATTICEPOINT, int L, double sigma, double theta, double rho1, double mu1, std::ofstream &file)
{
    std::vector<std::vector<int>> soil_lattice = init_lattice(L);

    int i = 0; // indexing for recording steps

    for (int step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            update(soil_lattice, L, sigma, theta, rho1, mu1);
        }
        int counts[5] = {0};
        for (const auto &row : soil_lattice)
        {
            for (int cell : row)
            {
                ++counts[cell];
            }
        }
        double emptys = static_cast<double>(counts[EMPTY]) / (L * L);
        double nutrients = static_cast<double>(counts[NUTRIENT]) / (L * L);
        double soil = static_cast<double>(counts[SOIL]) / (L * L);
        double greens = static_cast<double>(counts[GREEN_WORM]) / (L * L);

        file << step << "," << emptys << "," << nutrients << "," << soil << "," << greens << "\n";
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / STEPS_PER_LATTICEPOINT * 100 << "%\r" << std::flush;
    }
}

int main(int argc, char *argv[])
{
    double sigma = SIGMA;
    double theta = THETA;
    if (argc > 1)
        sigma = std::stod(argv[1]);
    if (argc > 2)
        theta = std::stod(argv[2]);

    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    // filePathStream << exeDir << "\\outputs\\timeseries2D\\sigma_" << sigma << "_theta_" << theta << ".csv";
    filePathStream << exeDir << "\\outputs\\timeseriesWormCounts2D\\sigma_" << sigma << "_theta_" << theta << ".csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "step,emptys,nutrients,soil,greens\n";
    run_timeseries(STEPS_PER_LATTICEPOINT, L, sigma, theta, RHO1, MU1, file);
    file.close();

    return 0;
}