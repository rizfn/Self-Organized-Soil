#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <array>


constexpr int COORDINATION_NUMBER = 3;
constexpr double P = 0.271;
constexpr int N_STEPS = 1000;
constexpr int N_SIMULATIONS = 1000;
constexpr int GENERATIONS = 3;  // Number of generations
constexpr int calculate_N_NODES(int coordination_number, int generations)
{
    int result = 1;
    for (int i = 0; i < generations - 1; ++i)
    {
        result *= coordination_number - 1;
    }
    return 1 + coordination_number * result;
}
constexpr int N_NODES = calculate_N_NODES(COORDINATION_NUMBER, GENERATIONS);

std::pair<std::array<std::vector<int>, N_NODES>, std::array<int, N_NODES>> generate_bethe_lattice(int generations)
{
    std::array<std::vector<int>, N_NODES> adjacency_list;
    std::array<int, N_NODES> distanceSquared;
    int next_node = COORDINATION_NUMBER;

    // Create the root node
    for (int i = 1; i < COORDINATION_NUMBER + 1; ++i)
    {
        adjacency_list[0].push_back(i);
        adjacency_list[i].push_back(0);
        adjacency_list[i].push_back(i); // Node can infect itself
        distanceSquared[i] = 1;
    }

    // Create the nodes of each generation
    for (int gen = 2; gen < generations; ++gen)
    {
        int start_node = std::pow(COORDINATION_NUMBER, gen - 2) + 1;
        int end_node = std::pow(COORDINATION_NUMBER, gen - 1) + 1;

        for (int node = start_node; node < end_node; ++node)
        {
            for (int i = 0; i < COORDINATION_NUMBER - 1; ++i)
            {
                adjacency_list[node].push_back(next_node);
                adjacency_list[next_node].push_back(node);
                adjacency_list[next_node].push_back(next_node); // Node can infect itself
                distanceSquared[next_node] = gen * gen;
                next_node++;
            }
        }
    }

    // For the boundary nodes
    for (int node = std::pow(COORDINATION_NUMBER, generations - 2) + 1; node < N_NODES; ++node)
    {
        adjacency_list[node].push_back(node); // Node can infect itself
    }

    return {adjacency_list, distanceSquared};
}

std::vector<double> linspace(double start, double end, int num)
{
    std::vector<double> linspaced;
    double delta = (end - start) / (double(num) - 1);
    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);
    return linspaced;
}


void run(std::ofstream &file)
{
    int total_nodes = N_NODES;
    auto [adjacency_list, distanceSquared] = generate_bethe_lattice(GENERATIONS);
    std::vector<std::vector<double>> meanSquaredDistance(N_STEPS, std::vector<double>(N_SIMULATIONS, 0));
    std::vector<std::vector<double>> nFilled(N_STEPS, std::vector<double>(N_SIMULATIONS, 0));
    std::fill(nFilled[0].begin(), nFilled[0].end(), 1);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int sim = 0; sim < N_SIMULATIONS; ++sim)
    {
        std::array<int, N_NODES> nodeValues;
        std::fill(nodeValues.begin(), nodeValues.end(), 0);
        nodeValues[0] = 1;  // Infection starts at the root node

        for (int i = 1; i < N_STEPS; ++i)
        {
            std::array<int, N_NODES> newNodeValues;
            std::fill(newNodeValues.begin(), newNodeValues.end(), 0);

            for (int node = 0; node < N_NODES; ++node)
            {
                std::vector<int>& neighbours = adjacency_list[node];
                if (nodeValues[node])
                {
                    for (auto &neighbour : neighbours)
                    {
                        bool neighbourInfected = distribution(generator) < P;
                        if (neighbourInfected)
                        {
                            newNodeValues[neighbour] = 1;
                        }
                    }
                }
            }
            nodeValues = newNodeValues;
            int nActiveNodes = std::accumulate(nodeValues.begin(), nodeValues.end(), 0);
            nFilled[i][sim] = nActiveNodes;
            if (nActiveNodes == 0)
            {
                break;
            }

            std::vector<int> activeNodes;
            for (int j = 0; j < nodeValues.size(); ++j)
            {
                if (nodeValues[j])
                {
                    activeNodes.push_back(j);
                }
            }

            double total_squared_distance = 0;
            for (auto &node : activeNodes)
            {
                total_squared_distance += distanceSquared[node];
            }

            meanSquaredDistance[i][sim] = total_squared_distance / activeNodes.size();
        }
        std::cout << "Progress: " << std::setw(5) << std::fixed << std::setprecision(2) << (100.0 * sim) / N_SIMULATIONS << "%\r";
    }

    std::vector<double> survivalProb(N_STEPS, 0);
    for (int i = 0; i < N_STEPS; ++i)
    {
        survivalProb[i] = std::count_if(nFilled[i].begin(), nFilled[i].end(), [](double val)
                                        { return val > 0; }) /
                          (double)N_SIMULATIONS;
    }

    std::vector<double> avgFilled(N_STEPS, 0);
    for (int i = 0; i < N_STEPS; ++i)
    {
        avgFilled[i] = std::accumulate(nFilled[i].begin(), nFilled[i].end(), 0.0) / N_SIMULATIONS;
    }

    std::vector<double> avgMeanSquaredDistance(N_STEPS, 0);
    for (int i = 0; i < N_STEPS; ++i)
    {
        avgMeanSquaredDistance[i] = std::accumulate(meanSquaredDistance[i].begin(), meanSquaredDistance[i].end(), 0.0) / N_SIMULATIONS;
    }

    for (int i = 0; i < N_STEPS; ++i)
    {
        file << i << "," << survivalProb[i] << "," << avgFilled[i] << "," << avgMeanSquaredDistance[i] << "\n";
    }
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filename;
    filename << exeDir << "/outputs/bethe/p_" << P << "_steps_" << N_STEPS << "gens" << GENERATIONS << ".csv";

    std::ofstream file(filename.str());

    file << "step,survival_prob,avg_filled,avg_mean_squared_distance\n";

    run(file);

    file.close();

    return 0;
}
