#include <bits/stdc++.h>
#include <windows.h>

// #pragma GCC optimize("Ofast","inline","fast-math","unroll-loops","no-stack-protector")
// #pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

// static auto _ = []()
// {std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::random_device rd;
std::mt19937 gen(rd());


class Worm
{
public:
    double theta;
    double mu;
    double rho;

    Worm(double theta, double mu, double rho) : theta(theta), mu(mu), rho(rho) {}
};

// Type aliases
using WormOrInt = std::variant<int, Worm>;
using WormOrInt2D = std::vector<std::vector<WormOrInt>>;
using WormOrInt3D = std::vector<WormOrInt2D>;

// Define a struct for coordinates
struct Coordinate
{
    int x;
    int y;
    int z;
};

// Define constants
constexpr int STEPS_PER_LATTICEPOINT = 1000;
constexpr double SIGMA = 0.5;
constexpr int L = 50; // side length of the cubic lattice

constexpr int EMPTY = 0;
constexpr int NUTRIENT = 1;
constexpr int SOIL = 2;
constexpr int WORM = 3;

// Define distributions
std::uniform_int_distribution<> dis_direction(0, 1);
std::uniform_int_distribution<> dis_coordinate(0, 2);
std::uniform_int_distribution<> dis_site(0, 3);
std::uniform_int_distribution<> dis_l(0, L - 1);
std::uniform_real_distribution<> dis_real(0.0, 1.0);
std::uniform_real_distribution<> dis_theta(0.025, 0.05);
std::uniform_real_distribution<> dis_mu(0.1, 0.9);
std::uniform_real_distribution<> dis_rho(0.1, 0.9);

Coordinate get_random_neighbour(Coordinate c, int L)
{
    // choose a random coordinate to change
    int coord_changing = dis_coordinate(gen);
    // choose a random direction to change the coordinate
    int change = 2 * dis_direction(gen) - 1;
    // change the coordinate
    c.x = (coord_changing == 0) ? (c.x + change + L) % L : c.x;
    c.y = (coord_changing == 1) ? (c.y + change + L) % L : c.y;
    c.z = (coord_changing == 2) ? (c.z + change + L) % L : c.z;

    return c;
}

WormOrInt3D init_lattice(int L)
{
    WormOrInt3D soil_lattice(L, WormOrInt2D(L, std::vector<WormOrInt>(L)));

    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int k = 0; k < L; ++k)
            {
                if (dis_site(gen) == WORM)
                {
                    soil_lattice[i][j][k] = Worm(dis_theta(gen), dis_mu(gen), dis_rho(gen));
                }
                else
                {
                    soil_lattice[i][j][k] = dis_site(gen);
                }
            }
        }
    }

    return soil_lattice;
}


void update(WormOrInt3D &soil_lattice, int L, double sigma)
{
    // select a random site
    Coordinate site = {dis_l(gen), dis_l(gen), dis_l(gen)};

    if (std::holds_alternative<int>(soil_lattice[site.x][site.y][site.z]))
    {
        int cell = std::get<int>(soil_lattice[site.x][site.y][site.z]);
        if (cell == EMPTY || cell == NUTRIENT)
        {
            // choose a random neighbour
            Coordinate nbr = get_random_neighbour(site, L);
            if (std::holds_alternative<int>(soil_lattice[nbr.x][nbr.y][nbr.z]))
            { // if neighbour is soil
                if (std::get<int>(soil_lattice[nbr.x][nbr.y][nbr.z]) == SOIL)
                { // fill with soil-filling rate
                    if (dis_real(gen) < sigma)
                    {
                        soil_lattice[site.x][site.y][site.z] = SOIL;
                    }
                }
            }
        }
    }
    else if (std::holds_alternative<Worm>(soil_lattice[site.x][site.y][site.z]))
    { // worm
        Worm &worm = std::get<Worm>(soil_lattice[site.x][site.y][site.z]);
        // check for death
        if (dis_real(gen) < worm.theta)
        {
            soil_lattice[site.x][site.y][site.z] = EMPTY;
        }
        else
        {
            // move into a neighbour
            Coordinate new_site = get_random_neighbour(site, L);
            // check the value of the new site
            if (std::holds_alternative<int>(soil_lattice[new_site.x][new_site.y][new_site.z]))
            {
                int new_site_value = std::get<int>(soil_lattice[new_site.x][new_site.y][new_site.z]);
                // move the worm
                soil_lattice[new_site.x][new_site.y][new_site.z] = std::move(worm);
                soil_lattice[site.x][site.y][site.z] = EMPTY;
                // check if the new site is nutrient
                if (new_site_value == NUTRIENT)
                {
                    // reproduce behind you
                    if (dis_real(gen) < worm.rho)
                    {
                        soil_lattice[site.x][site.y][site.z] = Worm(dis_theta(gen), dis_mu(gen), dis_rho(gen)); // new worm with new mu and rho
                        // double new_theta = dis_theta(gen);
                        // double new_mu = 2 * (dis_real(gen) - 0.5) * 0.2 + worm.mu;
                        // new_mu = std::max(0.1, std::min(new_mu, 0.9));
                        // double new_rho = 2 * (dis_real(gen) - 0.5) * 0.2 + worm.rho;
                        // new_rho = std::max(0.1, std::min(new_rho, 0.9));
                        // soil_lattice[site.x][site.y][site.z] = Worm(new_theta, new_mu, new_rho); // new worm with genetically mutated
                    }
                }
                // check if the new site is soil
                else if (new_site_value == SOIL)
                {
                    // leave nutrient behind
                    if (dis_real(gen) < worm.mu)
                    {
                        soil_lattice[site.x][site.y][site.z] = NUTRIENT;
                    }
                }
            }
            else if (std::holds_alternative<Worm>(soil_lattice[new_site.x][new_site.y][new_site.z]))
            {
                // swap the worms
                Worm &new_worm = std::get<Worm>(soil_lattice[new_site.x][new_site.y][new_site.z]);
                soil_lattice[new_site.x][new_site.y][new_site.z] = std::move(worm);
                soil_lattice[site.x][site.y][site.z] = std::move(new_worm);
            }
        }
    }
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
run_timeseries(int n_steps, int L, double sigma, std::vector<int> steps_to_record = {100, 1000, 10000, 100000})
{
    WormOrInt3D soil_lattice = init_lattice(L);

    std::vector<double> emptys(steps_to_record.size(), 0);
    std::vector<double> nutrients(steps_to_record.size(), 0);
    std::vector<double> soil(steps_to_record.size(), 0);
    std::vector<double> worms(steps_to_record.size(), 0);
    int i = 0; // indexing for recording steps

    for (int step = 0; step <= n_steps; ++step)
    {
        update(soil_lattice, L, sigma);
        if (i < steps_to_record.size() && step == steps_to_record[i])
        {
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / n_steps * 100 << "%\r" << std::flush;
            int counts[4] = {0};
            int worm_count = 0;
            for (const auto &layer : soil_lattice)
            {
                for (const auto &row : layer)
                {
                    for (const auto &cell : row)
                    {
                        if (std::holds_alternative<int>(cell))
                        {
                            ++counts[std::get<int>(cell)];
                        }
                        else if (std::holds_alternative<Worm>(cell))
                        {
                            ++worm_count;
                        }
                    }
                }
            }
            emptys[i] = counts[EMPTY];
            nutrients[i] = counts[NUTRIENT];
            soil[i] = counts[SOIL];
            worms[i] = worm_count;
            ++i;
        }
    }

    for (int j = 0; j < steps_to_record.size(); ++j)
    {
        emptys[j] /= L * L * L;
        nutrients[j] /= L * L * L;
        soil[j] /= L * L * L;
        worms[j] /= L * L * L;
    }

    return std::make_tuple(emptys, nutrients, soil, worms);
}

int main()
{
    // initialize the parameters
    int n_steps = STEPS_PER_LATTICEPOINT * L * L * L; // 3D

    // Create steps_to_record vector
    std::vector<int> steps_to_record;
    for (int i = 0; i <= n_steps; i += L * L * L)
    {
        steps_to_record.push_back(i);
    }

    // Run the time series
    auto [emptys, nutrients, soil, worms] = run_timeseries(n_steps, L, SIGMA, steps_to_record);

    // Normalize steps_to_record
    for (auto &step : steps_to_record)
    {
        step /= L * L * L;
    }

    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::string filePath = exeDir + "/outputs/test_mutations_3D_timeseries.csv";

    std::ofstream file;
    file.open(filePath);
    file << "step,emptys,nutrients,soil,worms\n";
    for (int i = 0; i < steps_to_record.size(); ++i)
    {
        file << steps_to_record[i] << "," << emptys[i] << "," << nutrients[i] << "," << soil[i] << "," << worms[i] << "\n";
    }
    file.close();

    return 0;
}
