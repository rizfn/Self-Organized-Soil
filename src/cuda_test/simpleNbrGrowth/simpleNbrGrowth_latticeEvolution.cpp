#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

constexpr double SIGMA = 1;
constexpr double THETA = 0.605;
constexpr int L = 100;
constexpr int N_STEPS = 500;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, L *L - 1);
std::uniform_int_distribution<> dis_dir(0, 3);
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<bool> initLattice()
{
    std::vector<bool> soil_lattice(L * L, false); // All sites off
    soil_lattice[L * L / 2 + L / 2] = true;       // Middle site on
    soil_lattice[L * L / 2 + (L / 2 + 1)] = true;
    soil_lattice[L * (L / 2 + 1) + L / 2] = true;
    soil_lattice[L * (L / 2 + 1) + (L / 2 + 1)] = true;
    return soil_lattice;
}


struct Coordinate
{
    int x;
    int y;
};

Coordinate get_random_neighbour(Coordinate site)
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
        // make it empty with rate theta
        if (dis_prob(gen) < THETA)
        {
            lattice[site_index] = false;
        }
    }
    else // if the site is empty
    {
        // choose a random neighbour
        Coordinate nbr = get_random_neighbour(site);
        int nbr_index = nbr.y * L + nbr.x;

        // if the neighbour is occupied, make it occupied with rate SIGMA
        if (lattice[nbr_index] && dis_prob(gen) < SIGMA)
        {
            lattice[site_index] = true;
        }
    }
}


void run_with_resetting(std::ofstream &file)
{
    std::vector<bool> lattice = initLattice();
    int n_resets = 0;
    int step = 0;
    while (step <= N_STEPS)
    {
        for (int i = 0; i < L * L; ++i)
        {
            updateLattice(lattice);
        }

        // Count the number of true sites
        int count = std::count(lattice.begin(), lattice.end(), true);

        // If count is 0, reset the simulation and continue
        if (count == 0)
        {
            std::cout << "Resetting simulation " << n_resets << "\r";
            ++n_resets;
            lattice = initLattice();
            step = 0;  // Reset the step counter
            file.clear();  // Clear the file for new data
            file.seekp(0);  // Move the file pointer to the beginning
            continue;
        }

        // Write the state of the lattice to the file
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                file << lattice[i * L + j];
                if (j != L - 1)
                {
                    file << ",";
                }
            }
            file << "\n";
        }
        file << "\n"; // Add an extra newline to separate timesteps

        ++step; // Increment the step counter
    }
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filename;
    filename << exeDir << "/outputs/latticeEvolution2D/sigma_" << SIGMA << "_theta_" << THETA << "_L_" << L << ".csv";

    std::ofstream file(filename.str());

    run_with_resetting(file);

    file.close();

    return 0;
}