#include <random>
#include <vector>
#include <thread>
#include <array>
#include <unordered_map>
#include <mutex>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::mutex mtx;                                            // for progress output
int max_threads = std::thread::hardware_concurrency() - 2; // Keep 2 threads free
int active_threads = 0;
int completed_threads = 0;

std::random_device rd;
std::mt19937 gen(rd());

// Define constants
constexpr float P = 0.592746;
constexpr int L = 4096; // 2^10 = 1024
constexpr int N_TRIALS = 10;
template<int N>
constexpr int log2()
{
    return (N < 2) ? 0 : 1 + log2<N / 2>();
}
constexpr int N_BOX_SIZES = log2<L>() - 1;


thread_local std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<bool> initLattice()
{
    std::vector<bool> soil_lattice(L * L, false); // Initialize all cells to false
    for (int i = 0; i < L * L; ++i)
    {
        soil_lattice[i] = dis_prob(gen) < P;
    }
    return soil_lattice;
}


class UnionFind
{
public:
    UnionFind(int n) : parent(n), rank(n, 0)
    {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int i)
    {
        if (parent[i] != i)
            parent[i] = find(parent[i]);
        return parent[i];
    }

    void union_set(int i, int j)
    {
        int ri = find(i), rj = find(j);
        if (ri != rj)
        {
            if (rank[ri] < rank[rj])
                parent[ri] = rj;
            else if (rank[ri] > rank[rj])
                parent[rj] = ri;
            else
            {
                parent[ri] = rj;
                ++rank[rj];
            }
        }
    }

private:
    std::vector<int> parent, rank;
};

void boxCount2D(std::vector<bool> &lattice, std::vector<unsigned int> &boxCounts)
{
    unsigned int s = 2;
    unsigned int size = L;
    unsigned char ni = 0;

    while (size > 2)
    {
        int sm = s >> 1; // s/2
        unsigned long im;
        unsigned long ismm;

        boxCounts[ni] = 0;
        for (unsigned long i = 0; i < (L - 1); i += s)
        {
            im = i * L;
            ismm = (i + sm) * L;
            for (unsigned long j = 0; j < (L - 1); j += s)
            {
                lattice[im + j] = lattice[im + j] || lattice[im + (j + sm)] || lattice[ismm + j] || lattice[ismm + (j + sm)];
                boxCounts[ni] += lattice[im + j];
            }
        }
        ni++;
        s <<= 1;    // s *= 2;
        size >>= 1; // size /= 2;
    }
}

std::vector<unsigned int> get_fractal_dimension(const std::vector<bool> &lattice)
{
    UnionFind uf_filled(L * L);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            int index = i * L + j;
            if (lattice[index])
            {
                if (lattice[((i - 1 + L) % L) * L + j])
                    uf_filled.union_set(index, ((i - 1 + L) % L) * L + j);
                if (lattice[i * L + (j - 1 + L) % L])
                    uf_filled.union_set(index, i * L + (j - 1 + L) % L);
                if (lattice[((i + 1) % L) * L + j])
                    uf_filled.union_set(index, ((i + 1) % L) * L + j);
                if (lattice[i * L + (j + 1) % L])
                    uf_filled.union_set(index, i * L + (j + 1) % L);
            }
        }
    }

    std::unordered_map<int, int> cluster_sizes_filled;
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            int index = i * L + j;
            if (lattice[index])
            {
                int root = uf_filled.find(index);
                ++cluster_sizes_filled[root];
            }
        }
    }

    int max_cluster_size = 0;
    int max_cluster_root = -1;
    for (const auto &pair : cluster_sizes_filled)
    {
        if (pair.second > max_cluster_size)
        {
            max_cluster_size = pair.second;
            max_cluster_root = pair.first;
        }
    }

    std::vector<bool> max_cluster_mask(L * L, false);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            int index = i * L + j;
            if (lattice[index] && uf_filled.find(index) == max_cluster_root)
            {
                max_cluster_mask[index] = true;
            }
        }
    }

    std::vector<unsigned int> box_counts(N_BOX_SIZES, 0);
    boxCount2D(max_cluster_mask, box_counts);

    return box_counts;

}


void run_fractalDim(std::ofstream &file)
{
    std::vector<unsigned int> box_lengths(N_BOX_SIZES);
    box_lengths[0] = 2;
    for (size_t i = 1; i < N_BOX_SIZES; ++i) {
        box_lengths[i] = box_lengths[i - 1] * 2;
    }

    for (int i = 0; i < N_TRIALS; ++i)
    {
        std::vector<bool> lattice = initLattice();
        std::vector<unsigned int> box_counts = get_fractal_dimension(lattice);
        
        for (size_t i = 0; i < N_BOX_SIZES; ++i)
        {
            file << box_lengths[i] << "\t" << box_counts[i] << std::endl;
        }
    }
}

int main(int argc, char *argv[]) {
    std::vector<std::thread> threads;

    for (int i = 0; i < N_TRIALS; ++i) {
        threads.push_back(std::thread([&, i] {
            std::string exePath = argv[0];
            std::string exeDir = std::filesystem::path(exePath).parent_path().string();

            std::ostringstream filename;
            filename << exeDir << "/outputs/fractalDim/p_" << P << "_L_" << L << "/" << i << ".tsv";

            std::ofstream file(filename.str());
            file << "box_length\tbox_count\n";

            run_fractalDim(file);

            file.close();

            // Lock the mutex before writing to the console
            std::lock_guard<std::mutex> lock(mtx);
            completed_threads++;
            std::cout << "Thread finished. Completion: " << (completed_threads * 100.0 / N_TRIALS) << "%\n";
        }));
    }

    for (std::thread &t : threads) {
        t.join();
    }

    return 0;
}