#include <random>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <tuple>  // Include the tuple header

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
constexpr int L = 2048; // side length of the square lattice
constexpr double RHO = 1;
constexpr double MU = 1;
constexpr double SIGMA = 0.1;
constexpr double THETA = 0.121;
constexpr int STEPS_PER_LATTICEPOINT = 6000;
constexpr int RECORDING_STEP = 3 * STEPS_PER_LATTICEPOINT / 4;
constexpr int RECORDING_INTERVAL = 20;

constexpr int EMPTY = 0;
constexpr int NUTRIENT = 1;
constexpr int SOIL = 2;
constexpr int GREEN_WORM = 3;

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

void update(std::vector<std::vector<int>> &soil_lattice)
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
            if (dis_real(gen) < SIGMA)
            {
                soil_lattice[site.x][site.y] = SOIL;
            }
        }
    }
    else if (soil_lattice[site.x][site.y] == GREEN_WORM)
    { // green worm
        // check for death
        if (dis_real(gen) < THETA)
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
                if (dis_real(gen) < RHO)
                {
                    soil_lattice[site.x][site.y] = GREEN_WORM;
                }
            }
            // check if the new site is soil
            else if (new_site_value == SOIL)
            {
                // leave nutrient behind
                if (dis_real(gen) < MU)
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

std::tuple<int, int, int> count_soil_boundaries(const std::vector<std::vector<int>> &soil_lattice)
{
    int soil_nonsoil_boundaries = 0;
    int soil_worm_boundaries = 0;
    int soil_nutrient_boundaries = 0;

    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if (soil_lattice[i][j] == SOIL)
            {
                // Check right neighbor
                if (soil_lattice[i][(j + 1) % L] != SOIL)
                {
                    soil_nonsoil_boundaries++;
                    if (soil_lattice[i][(j + 1) % L] == GREEN_WORM)
                    {
                        soil_worm_boundaries++;
                    }
                    else if (soil_lattice[i][(j + 1) % L] == NUTRIENT)
                    {
                        soil_nutrient_boundaries++;
                    }
                }

                // Check bottom neighbor
                if (soil_lattice[(i + 1) % L][j] != SOIL)
                {
                    soil_nonsoil_boundaries++;
                    if (soil_lattice[(i + 1) % L][j] == GREEN_WORM)
                    {
                        soil_worm_boundaries++;
                    }
                    else if (soil_lattice[(i + 1) % L][j] == NUTRIENT)
                    {
                        soil_nutrient_boundaries++;
                    }
                }
            }
        }
    }

    return std::make_tuple(soil_nonsoil_boundaries, soil_worm_boundaries, soil_nutrient_boundaries);
}

void run_csd(std::ofstream &file)
{
    std::vector<std::vector<int>> soil_lattice = init_lattice(L);

    for (int step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            update(soil_lattice);
        }

        if ((step >= RECORDING_STEP) && (step % RECORDING_INTERVAL == 0))
        {
            int soil_nonsoil_boundaries, soil_worm_boundaries, soil_nutrient_boundaries;
            std::tie(soil_nonsoil_boundaries, soil_worm_boundaries, soil_nutrient_boundaries) = count_soil_boundaries(soil_lattice);

            file << step << "\t" << soil_nonsoil_boundaries << "\t" << soil_worm_boundaries << "\t" << soil_nutrient_boundaries << "\n";
        }

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

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "\\outputs\\soil_boundaries\\sigma_" << sigma << "_theta_" << theta << ".tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "step\tsoil_nonsoil_boundaries\tsoil_worm_boundaries\tsoil_nutrient_boundaries\n";

    run_csd(file);

    file.close();

    return 0;
}