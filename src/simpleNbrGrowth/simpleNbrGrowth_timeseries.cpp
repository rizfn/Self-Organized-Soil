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
constexpr double THETA = 0.274;
constexpr int L = 1024; // 2^10 = 1024
constexpr int STEPS_PER_LATTICEPOINT = 4000;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, L *L - 1);
std::uniform_int_distribution<> dis_dir(0, 3);
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<bool> initLattice(int L)
{

    // std::vector<bool> soil_lattice(L * L);
    // for (int i = 0; i < L * L; ++i)
    // {
    //     soil_lattice[i] = dis(gen);
    // }
    // return soil_lattice;

    std::vector<bool> soil_lattice(L * L, false); // All sites off
    soil_lattice[L * L / 2 + L / 2] = true;       // Middle site on
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

void updateLattice(std::vector<bool> &lattice, double theta)
{
    // Choose a random site
    int site_index = dis_site(gen);
    Coordinate site = {site_index % L, site_index / L};

    if (lattice[site_index]) // if the site is occupied
    {
        // make it empty with rate theta
        if (dis_prob(gen) < theta)
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

void run_timeseries(std::ofstream &file)
{
    std::vector<bool> lattice = initLattice(L);

    for (int step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            updateLattice(lattice, THETA);
        }
        int occupied = 0;
        for (int i = 0; i < L * L; ++i)
        {
            if (lattice[i])
            {
                occupied++;
            }
        }
        file << static_cast<double>(occupied) / (L * L) << "\n" << std::flush;
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / STEPS_PER_LATTICEPOINT * 100 << "%\r" << std::flush;
    }
}

int main(int argc, char *argv[])
{
    std::vector<bool> lattice = initLattice(L);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filename;
    // filename << exeDir << "/outputs/timeseries/randomICs/sigma_" << SIGMA << "_theta_" << THETA << ".csv";
    filename << exeDir << "/outputs/timeseries/seedIC/sigma_" << SIGMA << "_theta_" << THETA << ".csv";

    std::ofstream file(filename.str());

    run_timeseries(file);

    file.close();

    return 0;
}