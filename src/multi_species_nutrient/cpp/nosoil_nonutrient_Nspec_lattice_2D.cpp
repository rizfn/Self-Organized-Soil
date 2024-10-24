#include <bits/stdc++.h>
#include <windows.h>

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
constexpr int STEPS_PER_LATTICEPOINT = 5000;
constexpr double RHO = 1;
constexpr int L = 500; // side length of the square lattice
constexpr long long N_STEPS = static_cast<long long>(STEPS_PER_LATTICEPOINT) * L * L;
constexpr int FINAL_STEPS_TO_RECORD = 500;

constexpr int N = 5; // number of species
constexpr std::array<int, N> WORMS = []
{
    std::array<int, N> arr{};
    for (int i = 0; i < N; ++i)
    {
        arr[i] = i;
    }
    return arr;
}();

// Define distributions
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, N-1);
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

void update(std::vector<std::vector<int>> &soil_lattice, int L, double rho)
{
    // select a random site
    Coordinate site = {dis_l(gen), dis_l(gen)};

    // get the worm at the selected site
    int worm = soil_lattice[site.x][site.y];

    // move into a neighbour
    Coordinate new_site = get_random_neighbour(site, L);

    // check the value of the new site
    int new_site_value = soil_lattice[new_site.x][new_site.y];

    // move the worm
    soil_lattice[new_site.x][new_site.y] = worm;
    soil_lattice[site.x][site.y] = new_site_value;

    // check if the new site is a worm that this worm can consume
    if (new_site_value == (worm + 1) % N) // check for worm to the right (with wrapping)
    {
        // reproduce behind you
        if (dis_real(gen) < rho)
        {
            soil_lattice[site.x][site.y] = worm;
        }
    }
}

void run(long long N_STEPS, int L, double rho, std::ofstream &file)
{
    std::vector<std::vector<int>> soil_lattice = init_lattice(L);

    int i = 0; // indexing for recording steps
    long long recordingStep = N_STEPS - (FINAL_STEPS_TO_RECORD * L * L);

    for (long long step = 0; step <= N_STEPS; ++step)
    {
        update(soil_lattice, L, rho);
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
    filePathStream << exeDir << "\\..\\outputs\\lattice2D\\" << N << "spec\\noSoilnoNutrient.tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "step\tlattice";
    file << "\n";
    run(N_STEPS, L, RHO, file);
    file.close();

    return 0;
}