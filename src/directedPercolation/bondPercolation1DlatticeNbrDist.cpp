#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

constexpr double P = 0.6447;
constexpr int L = 16384;
constexpr int RECORDING_STEP = (L * L) / 4; // spreading from center, in t^2 steps will spread distance 't'
constexpr int RECORDING_SKIP = 10;
constexpr int N_STEPS = RECORDING_STEP + RECORDING_SKIP*1000;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<bool> initLattice()
{
    std::vector<bool> soil_lattice(L, false);
    for (int i = 0; i < L; ++i)
    {
        if (i % 2 != 0)
        {
            soil_lattice[i] = true;
        }
    }
    return soil_lattice;
}

void updateLattice(std::vector<bool> &lattice)
{
    std::vector<bool> newLattice(L, false);
    for (int i = 0; i < L; ++i)
    {
        if (lattice[i]) // if the site is active
        {
            int left = (i - 1 + L) % L;
            int right = (i + 1) % L;
            if (dis_prob(gen) < P)
            {
                newLattice[left] = true;
            }
            if (dis_prob(gen) < P)
            {
                newLattice[right] = true;
            }
        }
    }
    lattice = newLattice;
}

void run_with_resetting(std::ofstream &file)
{
    std::vector<bool> lattice = initLattice();
    int n_resets = 0;
    int step = 0;
    while (step <= N_STEPS)
    {
        updateLattice(lattice);

        if (step % RECORDING_SKIP == 0)
        {
            // Count the number of true sites
            int count = std::count(lattice.begin(), lattice.end(), true);

            // If count is 0, reset the simulation and continue
            if (count == 0)
            {
                std::cout << "\t\t\t\tSim " << n_resets << " Reset\r";
                ++n_resets;
                lattice = initLattice();
                step = 0;      // Reset the step counter
                file.clear();  // Clear the file for new data
                file.seekp(0); // Move the file pointer to the beginning
                continue;
            }
            if (step >= RECORDING_STEP)
            {
                // Write the state of the lattice to the file
                for (int i = 0; i < L; ++i)
                {
                    file << lattice[i];
                    if (i != L - 1)
                    {
                        file << ",";
                    }
                }
                file << "\n";
            }

            std::cout << "Progress: " << std::setw(5) << std::fixed << std::setprecision(2) << (100.0 * step) / N_STEPS << "%\r";
        }
        ++step; // Increment the step counter
    }
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filename;
    filename << exeDir << "/outputs/latticeEvolution1D/nbrDist_p_" << P << "_L_" << L << ".csv";

    std::ofstream file(filename.str());

    run_with_resetting(file);

    file.close();

    return 0;
}