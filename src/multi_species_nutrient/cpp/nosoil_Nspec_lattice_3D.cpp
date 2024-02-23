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
    int z;
};

// Define constants
constexpr int STEPS_PER_LATTICEPOINT = 5000;
constexpr double THETA = 0.05;
constexpr double RHO = 1;
constexpr double MU = 1;
constexpr int L = 50; // side length of the square lattice
constexpr long long N_STEPS = static_cast<long long>(STEPS_PER_LATTICEPOINT) * L * L * L;
constexpr int FINAL_STEPS_TO_RECORD = 500;

constexpr int N = 4; // number of species
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
constexpr std::array<int, N> WORMS = []
{
    std::array<int, N> arr{};
    for (int i = 0; i < N; ++i)
    {
        arr[i] = i + N + 1;
    }
    return arr;
}();

// Define distributions
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_coord(0, 2);
std::uniform_int_distribution<> dis_site(0, 2 * N);
std::uniform_int_distribution<> dis_l(0, L - 1);
std::uniform_real_distribution<> dis_real(0.0, 1.0);

Coordinate get_random_neighbour(Coordinate c, int L)
{
    // choose a random coordinate to change
    int coord_changing = dis_coord(gen);
    // choose a random direction to change the coordinate
    int change = 2 * dis(gen) - 1;
    // change the coordinate
    c.x = (coord_changing == 0) ? (c.x + change + L) % L : c.x;
    c.y = (coord_changing == 1) ? (c.y + change + L) % L : c.y;
    c.z = (coord_changing == 2) ? (c.z + change + L) % L : c.z;

    return c;
}

std::vector<std::vector<std::vector<int>>> init_lattice(int L)
{
    std::vector<std::vector<std::vector<int>>> soil_lattice(L, std::vector<std::vector<int>>(L, std::vector<int>(L))); // Changed to 3D vector
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int k = 0; k < L; ++k)
            {
                soil_lattice[i][j][k] = dis_site(gen);
            }
        }
    }
    return soil_lattice;
}

void update(std::vector<std::vector<std::vector<int>>> &soil_lattice, int L, double theta, double rho, double mu)
{
    // select a random site
    Coordinate site = {dis_l(gen), dis_l(gen), dis_l(gen)};

    for (int i = 0; i < N; ++i)
    {
        if (soil_lattice[site.x][site.y][site.z] == WORMS[i])
        { // worm of species i
            // check for death
            if (dis_real(gen) < theta)
            {
                soil_lattice[site.x][site.y][site.z] = EMPTY;
            }
            else
            {
                // move into a neighbour
                Coordinate new_site = get_random_neighbour(site, L);
                // check the value of the new site
                int new_site_value = soil_lattice[new_site.x][new_site.y][new_site.z];
                // move the worm
                soil_lattice[new_site.x][new_site.y][new_site.z] = WORMS[i];
                soil_lattice[site.x][site.y][site.z] = EMPTY;
                // check if the new site is a nutrient that this worm can consume
                if (new_site_value == NUTRIENTS[(i + 1) % N]) // Modified condition to check for nutrient to the right (with wrapping)
                {
                    // reproduce behind you
                    if (dis_real(gen) < rho)
                    {
                        soil_lattice[site.x][site.y][site.z] = WORMS[i];
                    }
                }
                // check if the new site is empty
                else if (new_site_value == EMPTY)
                {
                    // leave nutrient behind
                    if (dis_real(gen) < mu)
                    {
                        soil_lattice[site.x][site.y][site.z] = NUTRIENTS[i];
                    }
                }
                // check if the new site is a worm
                else if (std::find(WORMS.begin(), WORMS.end(), new_site_value) != WORMS.end())
                {
                    // keep both with worms (undo the vacant space in original site)
                    soil_lattice[site.x][site.y][site.z] = new_site_value;
                }
            }
        }
    }
}

void run(long long N_STEPS, int L, double theta, double rho, double mu, std::ofstream &file)
{
    std::vector<std::vector<std::vector<int>>> soil_lattice = init_lattice(L);

    int i = 0;                                                               // indexing for recording steps
    long long recordingStep = N_STEPS - (FINAL_STEPS_TO_RECORD * L * L * L); // Changed to L^3

    for (long long step = 0; step <= N_STEPS; ++step)
    {
        update(soil_lattice, L, theta, rho, mu);
        if (step % (L * L * L) == 0) // Changed to L^3
        {
            if (step >= recordingStep)
            {
                file << step / (L * L * L) << "\t["; // Use \t as separator, changed to L^3
                for (int matrix_index = 0; matrix_index < L; ++matrix_index)
                {
                    file << "[";
                    for (int row_index = 0; row_index < L; ++row_index)
                    {
                        file << "[";
                        for (int cell_index = 0; cell_index < L; ++cell_index)
                        {
                            file << soil_lattice[matrix_index][row_index][cell_index];
                            if (cell_index != L - 1) // Check if it's not the last element in the row
                            {
                                file << ",";
                            }
                        }
                        file << "]";
                        if (row_index != L - 1) // Check if it's not the last row
                        {
                            file << ",";
                        }
                    }
                    file << "]";
                    if (matrix_index != L - 1) // Check if it's not the last matrix
                    {
                        file << ",";
                    }
                }
                file << "]";
                if (step / (L * L * L) != N_STEPS / (L * L * L)) // Check if it's not the last step
                {
                    file << "\n";
                }
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
    filePathStream << exeDir << "\\..\\..\\..\\docs\\data\\multispec_nutrient\\" << N << "spec\\nosoil_theta_" << THETA << ".tsv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "step\tlattice";
    file << "\n";
    run(N_STEPS, L, THETA, RHO, MU, file);
    file.close();

    return 0;
}