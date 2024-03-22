#include <bits/stdc++.h>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::random_device rd;
std::mt19937 gen(rd());

std::mutex file_mutex;
std::mutex thread_mutex;
std::condition_variable cv;
int active_threads = 0;
int max_threads = std::thread::hardware_concurrency() - 2;

// Define a struct for coordinates
struct Coordinate
{
    int x;
    int y;
};

// Define constants
constexpr int STEPS_PER_LATTICEPOINT = 2000;
constexpr double SIGMA = 1;
constexpr double THETA = 0.1;
constexpr double RHO1 = 0.25;
constexpr double MU1 = 1;
constexpr double RHO2 = 0.5;
constexpr double MU2 = 0;
constexpr int L = 100; // side length of the square lattice
constexpr long long N_STEPS = STEPS_PER_LATTICEPOINT * L * L;

constexpr auto STEPS_TO_RECORD = []() {
    return std::array<long long, 4> {
        N_STEPS * 70 / 100,
        N_STEPS * 80 / 100,
        N_STEPS * 90 / 100,
        N_STEPS * 100 / 100
    };
}();

constexpr int EMPTY = 0;
constexpr int NUTRIENT = 1;
constexpr int SOIL = 2;
constexpr int GREEN_WORM = 3;
constexpr int BLUE_WORM = 4;

constexpr int totalThreads = 20 * 20; // change to constexpr
int completedThreads = 0;

// Define distributions
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, 4);
std::uniform_int_distribution<> dis_l(0, L - 1);
std::uniform_real_distribution<> dis_real(0.0, 1.0);

Coordinate getRandomNeighbour(Coordinate c, int L)
{
    // choose a random coordinate to change
    int coordChanging = dis(gen);
    // choose a random direction to change the coordinate
    int change = 2 * dis(gen) - 1;
    // change the coordinate
    c.x = (coordChanging == 0) ? (c.x + change + L) % L : c.x;
    c.y = (coordChanging == 1) ? (c.y + change + L) % L : c.y;

    return c;
}

std::vector<std::vector<int>> init_lattice(int L)
{
    std::vector<std::vector<int>> soil_lattice(L, std::vector<int>(L));
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            soil_lattice[i][j] = dis_site(gen);
        }
    }

    return soil_lattice;
}

void update(std::vector<std::vector<int>> &soil_lattice, int L, double sigma, double theta, double rho1, double rho2, double mu1, double mu2)
{
    // select a random site
    Coordinate site = {dis_l(gen), dis_l(gen)};

    if (soil_lattice[site.x][site.y] == EMPTY || soil_lattice[site.x][site.y] == NUTRIENT)
    { // empty or nutrient
        // choose a random neighbour
        Coordinate nbr = getRandomNeighbour(site, L);
        if (soil_lattice[nbr.x][nbr.y] == SOIL)
        { // if neighbour is soil
            // fill with soil-filling rate
            if (dis_real(gen) < sigma)
            {
                soil_lattice[site.x][site.y] = SOIL;
            }
        }
    }
    else if (soil_lattice[site.x][site.y] == GREEN_WORM)
    { // green worm
        // check for death
        if (dis_real(gen) < theta)
        {
            soil_lattice[site.x][site.y] = EMPTY;
        }
        else
        {
            // move into a neighbour
            Coordinate new_site = getRandomNeighbour(site, L);
            // check the value of the new site
            int new_site_value = soil_lattice[new_site.x][new_site.y];
            // move the worm
            soil_lattice[new_site.x][new_site.y] = GREEN_WORM;
            soil_lattice[site.x][site.y] = EMPTY;
            // check if the new site is nutrient
            if (new_site_value == NUTRIENT)
            {
                // reproduce behind you
                if (dis_real(gen) < rho1)
                {
                    soil_lattice[site.x][site.y] = GREEN_WORM;
                }
            }
            // check if the new site is soil
            else if (new_site_value == SOIL)
            {
                // leave nutrient behind
                if (dis_real(gen) < mu1)
                {
                    soil_lattice[site.x][site.y] = NUTRIENT;
                }
            }
            // check if the new site is a worm
            else if (new_site_value == GREEN_WORM || new_site_value == BLUE_WORM)
            {
                // keep both with worms (undo the vacant space in original site)
                soil_lattice[site.x][site.y] = new_site_value;
            }
        }
    }
    else if (soil_lattice[site.x][site.y] == BLUE_WORM)
    { // blue worm
        // check for death
        if (dis_real(gen) < theta)
        {
            soil_lattice[site.x][site.y] = EMPTY;
        }
        else
        {
            // move into a neighbour
            Coordinate new_site = getRandomNeighbour(site, L);
            // check the value of the new site
            int new_site_value = soil_lattice[new_site.x][new_site.y];
            // move the worm
            soil_lattice[new_site.x][new_site.y] = BLUE_WORM;
            soil_lattice[site.x][site.y] = EMPTY;
            // check if the new site is nutrient
            if (new_site_value == NUTRIENT)
            {
                // reproduce behind you
                if (dis_real(gen) < rho2)
                {
                    soil_lattice[site.x][site.y] = BLUE_WORM;
                }
            }
            // check if the new site is soil
            else if (new_site_value == SOIL)
            {
                // leave nutrient behind
                if (dis_real(gen) < mu2)
                {
                    soil_lattice[site.x][site.y] = NUTRIENT;
                }
            }
            // check if the new site is a worm
            else if (new_site_value == GREEN_WORM || new_site_value == BLUE_WORM)
            {
                // keep both with worms (undo the vacant space in original site)
                soil_lattice[site.x][site.y] = new_site_value;
            }
        }
    }
}

void run_and_write_results(double sigma, double theta, const std::string &filePath)
{
    std::vector<std::vector<int>> soil_lattice = init_lattice(L);

    std::string json_data;

    for (int step = 1; step <= N_STEPS; ++step)
    {
        update(soil_lattice, L, sigma, theta, RHO1, RHO2, MU1, MU2);
        if (std::find(STEPS_TO_RECORD.begin(), STEPS_TO_RECORD.end(), step) != STEPS_TO_RECORD.end())
        {
            json_data += "{\"step\":" + std::to_string(step) + ",\"sigma\":" + std::to_string(sigma) + ",\"theta\":" + std::to_string(theta) + ",\"soil_lattice\":[";
            for (int i = 0; i < L; ++i)
            {
                json_data += "[";
                for (int j = 0; j < L; ++j)
                {
                    json_data += std::to_string(soil_lattice[i][j]);
                    if (j != L - 1)
                    {
                        json_data += ",";
                    }
                }
                json_data += "]";
                if (i != L - 1)
                {
                    json_data += ",";
                }
            }
            json_data += "]},";
        }
    }

    std::ofstream file;
    {
        std::lock_guard<std::mutex> lock(file_mutex);
        file.open(filePath, std::ios_base::app);
        file << json_data;
        file.close();
    }

    {
        completedThreads++;
        std::lock_guard<std::mutex> lock(thread_mutex);
        --active_threads;
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(completedThreads) / totalThreads * 100 << "%\r" << std::flush;
    }

    cv.notify_one();
}

std::vector<double> generate_values(double start, double stop, int num_values)
{
    std::vector<double> values(num_values);
    double step = (stop - start) / (num_values - 1);
    for (int i = 0; i < num_values; ++i)
    {
        values[i] = start + i * step;
    }
    return values;
}

int main(int argc, char *argv[])
{
    std::vector<double> sigma_values = generate_values(0, 1.0, 20);
    std::vector<double> theta_values = generate_values(0.0, 0.1, 20);

    std::filesystem::path exePath = std::filesystem::path(argv[0]).parent_path();
    std::string exeDir = exePath.string();
    std::ostringstream filename;
    filename << exeDir << "/../../docs/data/twospec_samenutrient/lattice2D_L_" << L << "_rho1_" << RHO1 << "_rho2_" << RHO2 << ".json";
    std::string filePath = filename.str();
    std::cout << exeDir << std::endl;

    std::vector<std::thread> threads;

    {
        std::lock_guard<std::mutex> lock(file_mutex);
        std::ofstream file(filePath);
        file << "[";
        file.close();
    }

    for (double sigma : sigma_values)
    {
        for (double theta : theta_values)
        {
            std::unique_lock<std::mutex> lock(thread_mutex);
            cv.wait(lock, []
                    { return active_threads < max_threads; }); // Wait if max threads are active
            ++active_threads;

            threads.push_back(std::thread(run_and_write_results, sigma, theta, filePath));
        }
    }

    for (std::thread &t : threads)
    {
        t.join();
    }

    {
        std::lock_guard<std::mutex> lock(file_mutex);
        std::ifstream file(filePath);
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        // Remove the last character
        if (!content.empty())
        {
            content.pop_back();
        }

        // Append a closing bracket
        content += "]";

        std::ofstream outFile(filePath);
        outFile << content;
        outFile.close();
    }

    return 0;
}