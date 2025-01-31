#include <random>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <tuple>

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
constexpr int L = 1024; // side length of the square lattice
constexpr double RHO_1 = 0.125;
constexpr double RHO_2 = 1;
constexpr double SIGMA = 0.5;
constexpr double THETA = 0.01;
constexpr int STEPS_PER_LATTICEPOINT = 4000;
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

void update(std::vector<std::vector<int>> &soil_lattice, int L, double sigma, double theta, double rho1, double rho2,
            int *soil_production, int *empty_production, int *nutrient_production, int *worm_production, int *parasite_production)
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
                (*soil_production)++;
            }
        }
    }
    else if (soil_lattice[site.x][site.y] == WORM || soil_lattice[site.x][site.y] == PARASITE)
    { // worm or parasite
        // check for death
        if (dis_real(gen) < theta)
        {
            soil_lattice[site.x][site.y] = EMPTY;
            (*empty_production)++;
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
                    if (is_parasite)
                        (*parasite_production)++;
                    else
                        (*worm_production)++;
                }
                else
                {
                    (*empty_production)++;
                }
            }
            // check if the new site is soil
            else if (new_site_value == SOIL)
            {
                if (is_parasite)
                {
                    (*empty_production)++;
                }
                // leave nutrient behind
                if (!is_parasite)
                {
                    soil_lattice[site.x][site.y] = NUTRIENT;
                    (*nutrient_production)++;
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

std::tuple<int, int, int, int> count_soil_boundaries(const std::vector<std::vector<int>> &soil_lattice)
{
    int soil_nonsoil_boundaries = 0;
    int soil_worm_boundaries = 0;
    int soil_nutrient_boundaries = 0;
    int soil_parasite_boundaries = 0;

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
                    if (soil_lattice[i][(j + 1) % L] == WORM)
                    {
                        soil_worm_boundaries++;
                    }
                    else if (soil_lattice[i][(j + 1) % L] == NUTRIENT)
                    {
                        soil_nutrient_boundaries++;
                    }
                    else if (soil_lattice[i][(j + 1) % L] == PARASITE)
                    {
                        soil_parasite_boundaries++;
                    }
                }

                // Check bottom neighbor
                if (soil_lattice[(i + 1) % L][j] != SOIL)
                {
                    soil_nonsoil_boundaries++;
                    if (soil_lattice[(i + 1) % L][j] == WORM)
                    {
                        soil_worm_boundaries++;
                    }
                    else if (soil_lattice[(i + 1) % L][j] == NUTRIENT)
                    {
                        soil_nutrient_boundaries++;
                    }
                    else if (soil_lattice[(i + 1) % L][j] == PARASITE)
                    {
                        soil_parasite_boundaries++;
                    }
                }
            }
        }
    }

    return std::make_tuple(soil_nonsoil_boundaries, soil_worm_boundaries, soil_nutrient_boundaries, soil_parasite_boundaries);
}

void run_csd(std::ofstream &file, double sigma, double theta, double rho1, double rho_2)
{
    std::vector<std::vector<int>> soil_lattice = init_lattice(L);

    int soil_production = 0;
    int empty_production = 0;
    int nutrient_production = 0;
    int worm_production = 0;
    int parasite_production = 0;

    for (int step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            update(soil_lattice, L, sigma, theta, rho1, rho_2, &soil_production, &empty_production, &nutrient_production, &worm_production, &parasite_production);
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
        double worms = static_cast<double>(counts[WORM]) / (L * L);
        double parasites = static_cast<double>(counts[PARASITE]) / (L * L);

        double e_production = static_cast<double>(empty_production) / (L * L);
        double n_production = static_cast<double>(nutrient_production) / (L * L);
        double s_production = static_cast<double>(soil_production) / (L * L);
        double w_production = static_cast<double>(worm_production) / (L * L);
        double p_production = static_cast<double>(parasite_production) / (L * L);

        if ((step >= RECORDING_STEP) && (step % RECORDING_INTERVAL == 0))
        {
            int soil_nonsoil_boundaries, soil_worm_boundaries, soil_nutrient_boundaries, soil_parasite_boundaries;
            std::tie(soil_nonsoil_boundaries, soil_worm_boundaries, soil_nutrient_boundaries, soil_parasite_boundaries) = count_soil_boundaries(soil_lattice);

            file << step << "\t" << soil_nonsoil_boundaries << "\t" << soil_worm_boundaries << "\t" << soil_nutrient_boundaries << "\t"
                 << soil_parasite_boundaries << "\t"
                 << emptys << "\t" << nutrients << "\t" << soil << "\t" << worms << "\t" << parasites << "\t"
                 << e_production << "\t" << n_production << "\t" << s_production << "\t" << w_production << "\t" << p_production << "\n";
        }

        // Reset the production counters
        soil_production = 0;
        empty_production = 0;
        nutrient_production = 0;
        worm_production = 0;
        parasite_production = 0;

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
    filePathStream << exeDir << "\\outputs\\soil_boundaries\\sigma_" << sigma << "_theta_" << theta << "_rhofactor_" << rho2/rho1 << ".tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "step\tsoil_nonsoil_boundaries\tsoil_worm_boundaries\tsoil_nutrient_boundaries\tsoil_parasite_boundaries\t"
         << "emptys\tnutrients\tsoil\tworms\tparasites\t"
         << "empty_production\tnutrient_production\tsoil_production\tworm_production\tparasite_production\n";

    run_csd(file, sigma, theta, rho1, rho2);

    file.close();

    return 0;
}