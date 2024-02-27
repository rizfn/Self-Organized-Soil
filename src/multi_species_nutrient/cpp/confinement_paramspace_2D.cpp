#include <bits/stdc++.h>
#include <windows.h>

// #pragma GCC optimize("Ofast","inline","fast-math","unroll-loops","no-stack-protector")
#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::mutex file_mutex;
std::mutex thread_mutex;
std::condition_variable cv;
int max_threads = std::thread::hardware_concurrency() - 2; // Keep 2 threads free
int active_threads = 0;

thread_local std::random_device rd;
thread_local std::mt19937 gen(rd());

// Define a struct for coordinates
struct Coordinate
{
    int x;
    int y;
};

// Define constants
constexpr int STEPS_PER_LATTICEPOINT = 20000;
constexpr double SIGMA = 0.5;
constexpr double RHO = 1;
constexpr double MU = 1;
constexpr int N_STEPS_TO_RECORD = 500;
constexpr int L = 500; // side length of the cubic lattice
constexpr long long N_STEPS = static_cast<long long>(STEPS_PER_LATTICEPOINT) * L * L;

constexpr int EMPTY = 0;

thread_local std::uniform_int_distribution<> dis(0, 1);
thread_local std::uniform_int_distribution<> dis_l(0, L - 1);
thread_local std::uniform_real_distribution<> dis_real(0.0, 1.0);

Coordinate get_random_neighbour(Coordinate c, int L)
{
    // choose a random coordinate to change
    int coord_changing = dis(gen);
    // choose a random direction to change the coordinate
    int change = 2 * dis(gen) - 1;
    // change the coordinate
    c.x = (coord_changing == 0) ? (c.x + change + L) % L : c.x;
    c.y = (coord_changing == 1) ? (c.y + change + L) % L : c.y;

    return c;
}

std::vector<std::vector<int>> init_lattice(int L, int N)
{
    std::uniform_int_distribution<> dis_site(0, 2 * N + 1);

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

void update(std::vector<std::vector<int>> &soil_lattice, int L, double sigma, double theta, double rho, double mu, const std::vector<int> &NUTRIENTS, const std::vector<int> &WORMS, int SOIL, int EMPTY, int N)
{
    // select a random site
    Coordinate site = {dis_l(gen), dis_l(gen)};

    if (soil_lattice[site.x][site.y] == EMPTY || std::find(NUTRIENTS.begin(), NUTRIENTS.end(), soil_lattice[site.x][site.y]) != NUTRIENTS.end())
    { // empty or nutrient
        // choose a random neighbour
        Coordinate nbr = get_random_neighbour(site, L);
        if (soil_lattice[nbr.x][nbr.y] == SOIL)
        { // if neighbour is soil
            // fill with soil-filling rate
            if (dis_real(gen) < sigma)
            {
                soil_lattice[site.x][site.y] = SOIL;
            }
        }
    }
    else
    {
        for (int i = 0; i < N; ++i)
        {
            if (soil_lattice[site.x][site.y] == WORMS[i])
            { // worm of species i
                // check for death
                if (dis_real(gen) < theta)
                {
                    soil_lattice[site.x][site.y] = EMPTY;
                }
                else
                {
                    // move into a neighbour
                    Coordinate new_site = get_random_neighbour(site, L);
                    // check the value of the new site
                    int new_site_value = soil_lattice[new_site.x][new_site.y];
                    // move the worm
                    soil_lattice[new_site.x][new_site.y] = WORMS[i];
                    soil_lattice[site.x][site.y] = EMPTY;
                    // check if the new site is a nutrient that this worm can consume
                    if (new_site_value == NUTRIENTS[(i + 1) % N]) // Modified condition to check for nutrient to the right (with wrapping)
                    {
                        // reproduce behind you
                        if (dis_real(gen) < rho)
                        {
                            soil_lattice[site.x][site.y] = WORMS[i];
                        }
                    }
                    // check if the new site is soil
                    else if (new_site_value == SOIL)
                    {
                        // leave nutrient behind
                        if (dis_real(gen) < mu)
                        {
                            soil_lattice[site.x][site.y] = NUTRIENTS[i];
                        }
                    }
                    // check if the new site is a worm
                    else if (std::find(WORMS.begin(), WORMS.end(), new_site_value) != WORMS.end())
                    {
                        // keep both with worms (undo the vacant space in original site)
                        soil_lattice[site.x][site.y] = new_site_value;
                    }
                }
            }
        }
    }
}

double count_soil_fraction(const std::vector<std::vector<int>> &soil_lattice, int SOIL)
{
    int total = 0;
    int soil = 0;
    for (const auto &row : soil_lattice)
    {
        for (int cell : row)
        {
            if (cell == SOIL)
            {
                ++soil;
            }
            ++total;
        }
    }
    return static_cast<double>(soil) / total;
}

void run_and_write_results(int N, double theta, double sigma, const std::string &filePath)
{
    std::vector<int> NUTRIENTS(N);
    std::vector<int> WORMS(N);
    int SOIL = N + 1;
    for (int i = 0; i < N; ++i)
    {
        NUTRIENTS[i] = i + 1;
        WORMS[i] = i + N + 2;
    }
    std::vector<std::vector<int>> soil_lattice = init_lattice(L, N);
    std::vector<double> soil_fractions;
    for (long long step = 0; step <= N_STEPS; ++step)
    {
        update(soil_lattice, L, sigma, theta, RHO, MU, NUTRIENTS, WORMS, SOIL, EMPTY, N);
        if (step >= N_STEPS - N_STEPS_TO_RECORD)
        {
            soil_fractions.push_back(count_soil_fraction(soil_lattice, SOIL));
        }
    }

    double mean_soil_fraction = std::accumulate(soil_fractions.begin(), soil_fractions.end(), 0.0) / soil_fractions.size();
    double min_soil_fraction = *std::min_element(soil_fractions.begin(), soil_fractions.end());
    double max_soil_fraction = *std::max_element(soil_fractions.begin(), soil_fractions.end());

    std::lock_guard<std::mutex> file_lock(file_mutex);
    std::ofstream file(filePath, std::ios_base::app);
    file << N << "," << theta << "," << sigma << "," << mean_soil_fraction << "," << min_soil_fraction << "," << max_soil_fraction << "\n";

    std::lock_guard<std::mutex> thread_lock(thread_mutex);
    --active_threads;
    cv.notify_one(); // Notify that a thread has finished
}

std::vector<double> linspace(double start, double end, int num)
{
    std::vector<double> linspaced;

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // Ensure that end is included

    return linspaced;
}

int main()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "\\..\\outputs\\confinement\\soilfracs_2D_parallel.csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "N,theta,sigma,mean_soil_fraction,min_soil_fraction,max_soil_fraction\n";
    file.flush();

    std::map<int, std::pair<double, double>> theta_ranges = {
        {2, {0.037, 0.038}},
        {3, {0.0236, 0.0241}},
        {4, {0.0175, 0.01799}},
        {5, {0.0139, 0.0153}},
        {6, {0.0135, 0.0145}},
        {7, {0.0125, 0.01275}},
        {8, {0.0106, 0.0108}}};

    std::vector<std::thread> threads;
    for (int N = 2; N <= 8; ++N)
    {
        std::vector<double> theta_values = linspace(theta_ranges[N].first, theta_ranges[N].second, 10);
        for (double theta : theta_values)
        {
            std::unique_lock<std::mutex> lock(thread_mutex);
            cv.wait(lock, []
                    { return active_threads < max_threads; }); // Wait if max threads are active
            ++active_threads;

            threads.push_back(std::thread(run_and_write_results, N, theta, SIGMA, filePath));
        }
    }
    for (auto &thread : threads)
    {
        thread.join();
    }

    return 0;
}