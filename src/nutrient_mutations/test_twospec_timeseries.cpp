#include <bits/stdc++.h>
#include <windows.h>

// #pragma GCC optimize("Ofast","inline","fast-math","unroll-loops","no-stack-protector")
// #pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

// static auto _ = []()
// {std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::random_device rd;
std::mt19937 gen(rd());

// Define a struct for coordinates
struct Coordinate
{
    int x;
    int y;
};

// Define constants
constexpr int STEPS_PER_LATTICEPOINT = 1000;
constexpr double SIGMA = 0.5;
constexpr double THETA = 0.025;
constexpr double RHO1 = 0.5;
constexpr double MU1 = 0.5;
constexpr double RHO2 = 1;
constexpr double MU2 = 0;
constexpr int L = 250; // side length of the square lattice

constexpr int EMPTY = 0;
constexpr int NUTRIENT = 1;
constexpr int SOIL = 2;
constexpr int GREEN_WORM = 3;
constexpr int BLUE_WORM = 4;

// Define distributions
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, 4);
std::uniform_int_distribution<> dis_l(0, L - 1);
std::uniform_real_distribution<> dis_real(0.0, 1.0);

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
            Coordinate new_site = get_random_neighbour(site, L);
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
            Coordinate new_site = get_random_neighbour(site, L);
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

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
run_timeseries(int n_steps, int L, double sigma, double theta, double rho1, double rho2, double mu1, double mu2, std::vector<int> steps_to_record = {100, 1000, 10000, 100000})
{
    std::vector<std::vector<int>> soil_lattice = init_lattice(L);

    std::vector<double> emptys(steps_to_record.size(), 0);
    std::vector<double> nutrients(steps_to_record.size(), 0);
    std::vector<double> soil(steps_to_record.size(), 0);
    std::vector<double> greens(steps_to_record.size(), 0);
    std::vector<double> blues(steps_to_record.size(), 0);
    int i = 0; // indexing for recording steps

    for (int step = 0; step <= n_steps; ++step)
    {
        update(soil_lattice, L, sigma, theta, rho1, rho2, mu1, mu2);
        if (i < steps_to_record.size() && step == steps_to_record[i])
        {
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / n_steps * 100 << "%\r" << std::flush;
            int counts[5] = {0};
            for (const auto &row : soil_lattice)
            {
                for (int cell : row)
                {
                    ++counts[cell];
                }
            }
            emptys[i] = counts[EMPTY];
            nutrients[i] = counts[NUTRIENT];
            soil[i] = counts[SOIL];
            greens[i] = counts[GREEN_WORM];
            blues[i] = counts[BLUE_WORM];
            ++i;
        }
    }

    for (int j = 0; j < steps_to_record.size(); ++j)
    {
        emptys[j] /= L * L;
        nutrients[j] /= L * L;
        soil[j] /= L * L;
        greens[j] /= L * L;
        blues[j] /= L * L;
    }

    return std::make_tuple(emptys, nutrients, soil, greens, blues);
}

int main()
{
    // initialize the parameters
    int n_steps = STEPS_PER_LATTICEPOINT * L * L; // 2D

    // Create steps_to_record vector
    std::vector<int> steps_to_record;
    for (int i = 0; i <= n_steps; i += L * L)
    {
        steps_to_record.push_back(i);
    }

    // Run the time series
    auto [emptys, nutrients, soil, greens, blues] = run_timeseries(n_steps, L, SIGMA, THETA, RHO1, RHO2, MU1, MU2, steps_to_record);

    // Normalize steps_to_record
    for (auto &step : steps_to_record)
    {
        step /= L * L;
    }

    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::string filePath = exeDir + "\\outputs\\test_twospec_timeseries.csv";

    std::ofstream file;
    file.open(filePath);
    file << "step,emptys,nutrients,soil,greens,blues\n";
    for (int i = 0; i < steps_to_record.size(); ++i)
    {
        file << steps_to_record[i] << "," << emptys[i] << "," << nutrients[i] << "," << soil[i] << "," << greens[i] << "," << blues[i] << "\n";
    }
    file.close();

    return 0;
}
