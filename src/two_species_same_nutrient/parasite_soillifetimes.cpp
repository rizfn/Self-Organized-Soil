#include <bits/stdc++.h>
#include <windows.h>
#include <unordered_map>

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
    int x, y, z;
    bool operator==(const Coordinate &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

// For the hashmap to store soil lifetimes
struct CoordinateHash
{
    std::size_t operator()(const Coordinate &c) const
    {
        return std::hash<int>()(c.x) ^ std::hash<int>()(c.y) ^ std::hash<int>()(c.z);
    }
};

// Define constants
constexpr int STEPS_PER_LATTICEPOINT = 2000;
constexpr double SIGMA = 0.5;
constexpr double THETA = 0.025;
constexpr double RHO1 = 0.5;
constexpr double MU1 = 0.5;
constexpr double RHO2 = 1;
constexpr double MU2 = 0;
constexpr int L = 50; // side length of the cubic lattice

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

void update(std::vector<std::vector<std::vector<int>>> &soil_lattice, std::unordered_map<Coordinate, int, CoordinateHash> &soil_creation_times, int L, double sigma, double theta, double rho1, double rho2, double mu1, double mu2, std::ostringstream &buffer, int current_step)
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
                soil_creation_times[site] = current_step;
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
                int creation_time = soil_creation_times[site];
                int lifetime = current_step - creation_time;
                buffer << current_step << "," << lifetime << "\n";
                soil_creation_times.erase(site);
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
                int creation_time = soil_creation_times[site];
                int lifetime = current_step - creation_time;
                buffer << current_step << "," << lifetime << "\n";
                soil_creation_times.erase(site);
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

void run_lifetimes(int n_steps, int L, double sigma, double theta, double rho1, double rho2, double mu1, double mu2, std::ofstream &file)
{
    std::vector<std::vector<std::vector<int>>> soil_lattice = init_lattice(L); // 3D lattice
    std::unordered_map<Coordinate, int, CoordinateHash> soil_creation_times;   // 3D hashmap

    std::ostringstream buffer;

    for (int step = 0; step <= n_steps; ++step)
    {
        update(soil_lattice, soil_creation_times, L, sigma, theta, rho1, rho2, mu1, mu2, buffer, step);

        if (step % (L * L * L) == 0)
        {
            file << buffer.str();
            buffer.str(""); // clear the buffer
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / n_steps * 100 << "%\r" << std::flush;
        }
    }

    // write any remaining data in the buffer
    file << buffer.str();
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
    filename << exeDir << "\\outputs\\soil_lifetimes\\sigma_" << sigma << "_theta_" << theta << ".csv";
    std::string filePath = filename.str();

    std::ofstream file;
    file.open(filePath);

    // Write the header
    file << "step,soil_lifetime\n";

    // Run the time series
    run_lifetimes(n_steps, L, sigma, theta, RHO1, RHO2, MU1, MU2, file);

    file.close();

    return 0;
}