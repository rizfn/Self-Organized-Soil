#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>

constexpr int COORDINATION_NUMBER = 3;
constexpr double P = 0.271;
constexpr int N_STEPS = 1000;
constexpr int N_SIMULATIONS = 1000;

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

int get_distance_to_origin(int node, std::map<int, std::vector<int>> &adjacency_list)
{
    int distance = 0;
    while (node != 0)
    {
        node = *std::min_element(adjacency_list[node].begin(), adjacency_list[node].end());
        distance += 1;
    }
    return distance;
}

void run(std::ofstream &file)
{
    std::vector<std::vector<double>> meanSquaredDistance(N_STEPS, std::vector<double>(N_SIMULATIONS, 0));
    std::vector<std::vector<double>> nFilled(N_STEPS, std::vector<double>(N_SIMULATIONS, 0));
    std::fill(nFilled[0].begin(), nFilled[0].end(), 1);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int sim = 0; sim < N_SIMULATIONS; ++sim)
    {
        std::vector<int> nodeValues = {1};
        nodeValues.resize(COORDINATION_NUMBER + 1, 0);

        std::map<int, std::vector<int>> adjacencyList;
        for (int i = 0; i <= COORDINATION_NUMBER; ++i)
        {
            adjacencyList[i].push_back(i);
            if (i != 0)
            {
                adjacencyList[0].push_back(i);
                adjacencyList[i].push_back(0);
            }
        }

        int lastNode = COORDINATION_NUMBER;

        for (int i = 1; i < N_STEPS; ++i)
        {
            std::vector<int> newNodeValues(nodeValues.size(), 0);
            std::map<int, std::vector<int>> adjacencyListCopy = adjacencyList;
            int previousLastNode = lastNode;

            for (auto &[node, neighbours] : adjacencyList)
            {
                if (nodeValues[node])
                {
                    for (auto &neighbour : neighbours)
                    {
                        bool neighbourInfected = distribution(generator) < P;
                        if (neighbourInfected)
                        {
                            newNodeValues[neighbour] = 1;
                            if (adjacencyList[neighbour].size() == 2)
                            {
                                for (int j = 0; j < COORDINATION_NUMBER - 1; ++j)
                                {
                                    adjacencyListCopy[neighbour].push_back(++lastNode);
                                    adjacencyListCopy[lastNode] = {neighbour, lastNode};
                                }
                            }
                        }
                    }
                }
            }

            newNodeValues.resize(lastNode + 1, 0);
            adjacencyList = adjacencyListCopy;
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
                total_squared_distance += std::pow(get_distance_to_origin(node, adjacencyList), 2);
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
    filename << exeDir << "/outputs/bethe/p_" << P << "_steps_" << N_STEPS << ".csv";

    std::ofstream file(filename.str());

    file << "step,survival_prob,avg_filled,avg_mean_squared_distance\n";

    run(file);

    file.close();

    return 0;
}
