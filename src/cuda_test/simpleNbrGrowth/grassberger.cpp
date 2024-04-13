#include <random>
#include <vector>
#include <thread>
#include <array>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <functional>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::mutex mtx; // for progress output
std::condition_variable cv;
int max_threads = std::thread::hardware_concurrency() - 2; // Keep 2 threads free
int active_threads = 0;
int completed_threads = 0;

thread_local std::random_device rd;
thread_local std::mt19937 gen(rd());

// Define constants
constexpr double SIGMA = 0.5/2;
constexpr double THETA = 0.305/2;
constexpr int L = 1024; // 2^10 = 1024
constexpr int STEPS_PER_LATTICEPOINT = 1000;
constexpr int N_SIMULATIONS = 10000;
constexpr int SIMULATION_BATCHSIZE = 20;
constexpr int N_BATCHES = N_SIMULATIONS / SIMULATION_BATCHSIZE;

std::uniform_int_distribution<> dis_site(0, L *L - 1);
std::uniform_int_distribution<> dis_dir(0, 3);
std::uniform_real_distribution<> dis_prob(0, 1);

std::vector<long long> createDistanceSquaredArray(int L)
{
    std::vector<long long> distanceSquared(L * L);

    int centerX = L / 2;
    int centerY = L / 2;

    for (int y = 0; y < L; ++y)
    {
        for (int x = 0; x < L; ++x)
        {
            int dx = x - centerX;
            int dy = y - centerY;
            distanceSquared[x + y * L] = dx * dx + dy * dy;
        }
    }

    return distanceSquared;
}
std::vector<long long> distanceSquared = createDistanceSquaredArray(L);

std::vector<bool> initLattice(int L)
{
    std::vector<bool> soil_lattice(L * L, false); // Initialize all sites to false
    int centralIndex = L * (L / 2) + L / 2;       // Calculate the index of the central site
    soil_lattice[centralIndex] = true;            // Set the central site to true
    return soil_lattice;
}

struct Coordinate
{
    int x;
    int y;
};

class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i<threads; ++i)
            workers.emplace_back(
                [this] {
                    for(;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock,
                                [this]{ return this->stop || !this->tasks.empty(); });
                            if(this->stop && this->tasks.empty())
                                return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        task();
                    }
                }
            );
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // don't allow enqueueing after stopping the pool
            if(stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;
    
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

Coordinate get_random_neighbour(Coordinate site, int L)
{
    int dir = dis_dir(gen);
    switch (dir)
    {
    case 0: // left
        return {(site.x - 1 + L) % L, site.y};
    case 1: // right
        return {(site.x + 1) % L, site.y};
    case 2: // above
        return {site.x, (site.y - 1 + L) % L};
    case 3: // below
        return {site.x, (site.y + 1) % L};
    }
    return site; // should never reach here
}

void updateLattice(std::vector<bool> &lattice, double theta)
{
    // Choose a random site
    int site_index = dis_site(gen);
    Coordinate site = {site_index % L, site_index / L};

    if (lattice[site_index]) // if the site is occupied
    {
        // make it empty with rate theta
        if (dis_prob(gen) < theta)
        {
            lattice[site_index] = false;
        }
    }
    else // if the site is empty
    {
        // choose a random neighbour
        Coordinate nbr = get_random_neighbour(site, L);
        int nbr_index = nbr.y * L + nbr.x;

        // if the neighbour is occupied, make it occupied with rate SIGMA
        if (lattice[nbr_index] && dis_prob(gen) < SIGMA)
        {
            lattice[site_index] = true;
        }
    }
}

int countLattice(std::vector<bool> &lattice)
{
    int count = 0;
    for (bool site : lattice)
    {
        if (site)
        {
            count++;
        }
    }
    return count;
}

double calculateR2(std::vector<bool> &lattice, std::vector<long long> &distanceSquared, int activeSites)
{
    if (activeSites == 0)
    {
        return 0;
    }
    long long totalDistance = 0;
    for (int i = 0; i < L * L; ++i)
    {
        if (lattice[i])
        {
            totalDistance += distanceSquared[i];
        }
    }
    return static_cast<double>(totalDistance) / activeSites;
}

void run(std::ofstream &file, int simulationNumber)
{
    std::vector<bool> lattice = initLattice(L);
    int activeSites = countLattice(lattice);
    file << simulationNumber << "," << 0 << "," << activeSites << "," << calculateR2(lattice, distanceSquared, activeSites) << "\n";
    for (int step = 1; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            updateLattice(lattice, THETA);
        }

        int activeSites = countLattice(lattice);

        double r2 = calculateR2(lattice, distanceSquared, activeSites);

        // Write the data for this step
        file << simulationNumber << "," << step << "," << activeSites << "," << r2 << "\n";

        if (activeSites == 0)
        {
            break;
        }
    }
}

void run_batch(int batchNumber, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filename;
    filename << exeDir << "/outputs/grassberger/critical_point/sigma_" << SIGMA << "_theta_" << THETA << "_batch_" << batchNumber << ".csv";

    std::ofstream file(filename.str());

    if (file.is_open())
    {
        file << "simulation,time,activeCounts,R2\n";
        for (int i = 0; i < SIMULATION_BATCHSIZE; ++i)
        {
            int simulationNumber = batchNumber * SIMULATION_BATCHSIZE + i;
            run(file, simulationNumber);
        }
    }

    file.close();

    // Lock the mutex before writing to the console
    std::lock_guard<std::mutex> lock(mtx);
    completed_threads++;
    std::cout << "Progress: " << (completed_threads * 100.0 / N_BATCHES) << "%"
              << "\r" << std::flush;

    // Decrease the number of active threads and notify the condition variable
    active_threads--;
    cv.notify_one();
}

int main(int argc, char *argv[]) {
    ThreadPool pool(max_threads);

    for (int i = 0; i < N_BATCHES; ++i) {
        pool.enqueue(run_batch, i, argv);
    }

    return 0;
}
