#include <bits/stdc++.h>
#include <windows.h>

// FIX: BIG ERROR

// #pragma GCC optimize("Ofast","inline","fast-math","unroll-loops","no-stack-protector")
#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::random_device rd;
std::mt19937 gen(rd());

// Define a struct for coordinates
struct Coordinate
{
    int x;
    int y;
};

// Define constants
constexpr int STEPS_PER_LATTICEPOINT = 20000;
constexpr double SIGMA = 0.5;
constexpr double THETA = 0.0238;
constexpr double RHO = 1;
constexpr double MU = 1;
constexpr int L = 500; // side length of the square lattice
constexpr long long N_STEPS = static_cast<long long>(STEPS_PER_LATTICEPOINT) * L * L;
constexpr int FINAL_STEPS_TO_RECORD = 500;

constexpr int N = 3; // number of species
constexpr int EMPTY = 0;
constexpr std::array<int, N> NUTRIENTS = []
{
    std::array<int, N> arr{};
    for (int i = 0; i < N; ++i)
    {
        arr[i] = i + 1;
    }
    return arr;
}();
constexpr int SOIL = N + 1;
constexpr std::array<int, N> WORMS = []
{
    std::array<int, N> arr{};
    for (int i = 0; i < N; ++i)
    {
        arr[i] = i + N + 2;
    }
    return arr;
}();

// Define distributions
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, 2 * N + 1);
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
    std::vector<std::vector<int>> soil_lattice(L, std::vector<int>(L)); // Changed to 2D vector
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            soil_lattice[i][j] = dis_site(gen);
        }
    }
    return soil_lattice;
}

void update(std::vector<std::vector<int>> &soil_lattice, int L, double sigma, double theta, double rho, double mu)
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

void run(long long N_STEPS, int L, double sigma, double theta, double rho, double mu, std::ofstream &file)
{
    std::vector<std::vector<int>> soil_lattice = init_lattice(L);

    int i = 0; // indexing for recording steps
    long long recordingStep = N_STEPS - (FINAL_STEPS_TO_RECORD * L * L);

    for (long long step = 0; step <= N_STEPS; ++step)
    {
        update(soil_lattice, L, sigma, theta, rho, mu);
        if (step % (L * L) == 0)
        {
            if (step >= recordingStep)
            {
                file << step / (L * L) << "\t["; // Use \t as separator
                for (const auto &row : soil_lattice)
                {
                    file << "[";
                    for (int cell : row)
                    {
                        file << cell;
                        if (&cell != &row.back()) // Check if it's not the last element in the row
                        {
                            file << ",";
                        }
                    }
                    file << "]";
                    if (&row != &soil_lattice.back()) // Check if it's not the last row
                    {
                        file << ",";
                    }
                }
                file << "]\n";
            }
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / N_STEPS * 100 << "%\r" << std::flush;
        }
    }
}

int main()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "\\..\\outputs\\lattice2D\\" << N << "spec\\sigma_" << SIGMA << "_theta_" << THETA << ".tsv"; // Changed to 2D
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "step\tlattice";
    file << "\n";
    run(N_STEPS, L, SIGMA, THETA, RHO, MU, file);
    file.close();

    return 0;
}