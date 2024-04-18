#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

constexpr double P = 0.2873;
constexpr int L = 2048;
constexpr int N_STEPS = 2000;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<std::vector<bool>> initLattice()
{
    std::vector<std::vector<bool>> soil_lattice(L, std::vector<bool>(L, false));
    soil_lattice[L / 2][L / 2] = true;
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

void run_with_resetting(std::ofstream &file)
{
    std::vector<std::vector<bool>> lattice = initLattice();
    std::vector<std::vector<bool>> densestLattice = lattice;
    int densestCount = 1;
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
            step = 0;      // Reset the step counter
            file.clear();  // Clear the file for new data
            file.seekp(0); // Move the file pointer to the beginning
            continue;
        }

        if (count > densestCount)
        {
            densestCount = count;
            densestLattice = lattice;
        }

        ++step; // Increment the step counter
    }
    // Write the state of the lattice to the file
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            file << densestLattice[i][j];
            if (j != L - 1)
            {
                file << " ";
            }
        }
        file << "\n";
    }
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filename;
    filename << exeDir << "/outputs/lattice2D/survival_p_" << P << "_L_" << L << "_steps_" << N_STEPS << ".csv";

    std::ofstream file(filename.str());

    run_with_resetting(file);

    file.close();

    return 0;
}