#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

constexpr double P = 0.4;
constexpr int L = 100;
constexpr int N_STEPS = 200;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<bool> initLattice(int L)
{
    std::vector<bool> lattice(L, false); // Initialize with all zeros
    for (int i = 0; i < L; i += 20)
    {
        lattice[i] = true; // Set every 20th site to 1
    }
    return lattice;
}

void updateLattice(std::vector<bool> &lattice)
{
    std::vector<bool> newLattice(L, false);
    for (int site = 0; site < L; ++site)
    {
        if (lattice[site]) // if the site is active
        {
            int left = (site - 1 + L) % L;
            int right = (site + 1) % L;
            if (dis_prob(gen) < P)
            {
                newLattice[site] = true;
            }
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

void run(std::ofstream &file)
{
    std::vector<bool> lattice = initLattice(L);
    for (int step = 0; step <= N_STEPS; ++step)
    {
        updateLattice(lattice);
        for (const auto &cell : lattice)
        {
            file << cell << ",";
        }
        file << "\n";
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / N_STEPS * 100 << "%\r" << std::flush;
    }
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filename;
    filename << exeDir << "/outputs/timeseries1D/dirP_p_" << P << ".tsv";

    std::ofstream file(filename.str());

    run(file);

    file.close();

    return 0;
}