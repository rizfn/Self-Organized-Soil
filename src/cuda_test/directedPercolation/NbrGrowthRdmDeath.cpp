#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

constexpr double SIGMA = 1;
constexpr double THETA = 0.4;
constexpr int L = 100;
constexpr int N_STEPS = 200;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, L - 1);
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

int get_random_neighbour(int site, int L)
{
    int left = (site - 1 + L) % L;
    int right = (site + 1) % L;
    return dis(gen) ? left : right;
}

void updateLattice(std::vector<bool> &lattice)
{
    int site = dis_site(gen);
    if (lattice[site]) // if the site is occupied
    {
        if (dis_prob(gen) < THETA)
        {
            lattice[site] = false;
        }
    }
    else // if the site is empty
    {
        int nbr = get_random_neighbour(site, L);
        if (lattice[nbr] && dis_prob(gen) < SIGMA)
        {
            lattice[site] = true;
        }
    }
}

void run(std::ofstream &file)
{
    std::vector<bool> lattice = initLattice(L);
    for (int step = 0; step <= N_STEPS; ++step)
    {
        for (int i = 0; i < L; ++i)
        {
            updateLattice(lattice);
        }
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
    filename << exeDir << "/outputs/sigma_" << SIGMA << "_theta_" << THETA << ".tsv";

    std::ofstream file(filename.str());

    run(file);

    file.close();

    return 0;
}