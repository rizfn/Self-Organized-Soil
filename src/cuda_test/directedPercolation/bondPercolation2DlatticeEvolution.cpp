#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

constexpr double P = 0.2873;
constexpr int L = 100;
constexpr int N_STEPS = 500;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<std::vector<bool>> initLattice()
{
    std::vector<std::vector<bool>> soil_lattice(L, std::vector<bool>(L, false));
    soil_lattice[L / 2][L / 2] = true;
    soil_lattice[L / 2][(L / 2) + 1] = true;
    soil_lattice[(L / 2) + 1][L / 2] = true;
    soil_lattice[(L / 2) + 1][(L / 2) + 1] = true;
    return soil_lattice;
}

void updateLattice(std::vector<std::vector<bool>> &lattice)
{
    std::vector<std::vector<bool>> newLattice(L, std::vector<bool>(L, false));
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if (lattice[i][j]) // if the site is active
            {
                int left = (j - 1 + L) % L;
                int right = (j + 1) % L;
                int up = (i - 1 + L) % L;
                int down = (i + 1) % L;
                if (dis_prob(gen) < P)
                {
                    newLattice[i][left] = true;
                }
                if (dis_prob(gen) < P)
                {
                    newLattice[i][right] = true;
                }
                if (dis_prob(gen) < P)
                {
                    newLattice[up][j] = true;
                }
                if (dis_prob(gen) < P)
                {
                    newLattice[down][j] = true;
                }
            }
        }
    }
    lattice = newLattice;
}

// todo: write the 0th step
void run_with_resetting(std::ofstream &file)
{
    std::vector<std::vector<bool>> lattice = initLattice();
    int n_resets = 0;
    int step = 0;
    while (step <= N_STEPS)
    {
        updateLattice(lattice);

        // Count the number of true sites
        int count = 0;
        for (const auto &row : lattice)
        {
            count += std::count(row.begin(), row.end(), true);
        }

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
                file << lattice[i][j];
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
    filename << exeDir << "/outputs/latticeEvolution2D/p_" << P << "_L_" << L << ".csv";

    std::ofstream file(filename.str());

    run_with_resetting(file);

    file.close();

    return 0;
}