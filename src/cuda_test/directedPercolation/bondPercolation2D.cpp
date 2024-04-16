#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

constexpr double P = 0.3845;
constexpr int L = 500;
constexpr int N_STEPS = 4000;
constexpr float initialDensity = 1e-5;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<std::vector<bool>> initLattice()
{
    std::vector<std::vector<bool>> soil_lattice(L, std::vector<bool>(L));
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if (dis_prob(gen) < initialDensity)
            {
                soil_lattice[i][j] = true;
            }
        }
    }
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

void run(std::ofstream &file)
{
    std::vector<std::vector<bool>> lattice = initLattice();
    for (int step = 0; step <= N_STEPS; ++step)
    {
        updateLattice(lattice);
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / N_STEPS * 100 << "%\r" << std::flush;
    }
    for (size_t i = 0; i < L; ++i)
    {
        for (size_t j = 0; j < L; ++j)
        {
            file << lattice[i][j];
            if (j != L - 1)
            {
                file << ",";
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
    filename << exeDir << "/outputs/lattice2D/rho_" << initialDensity << "_p_" << P << "_L_" << L << ".csv";

    std::ofstream file(filename.str());

    run(file);

    file.close();

    return 0;
}