import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    n_steps = 100
    L = 1000
    array = np.zeros((L, L))
    array[:L//2, :L//2] = 1
    # array = np.random.rand(L, L) < 0.5
    print(array.shape)
    total_histogram = np.zeros(L+1)

    for i in tqdm(range(n_steps)):
        # choose random active point in array
        active_points = np.where(array == 1)
        n_active_points = len(active_points[0])
        idx = np.random.randint(n_active_points)
        x = active_points[0][idx]
        y = active_points[1][idx]

        # calculate manhattan distance from random point to all other points
        dist = -np.ones((L, L))
        for j in range(L):
            for k in range(L):
                if array[j, k] == 1:
                    dx = np.abs(x-j)
                    dy = np.abs(y-k)
                    dist[j, k] = min(dx, L-dx) + min(dy, L-dy)
        
        distance_histogram = np.zeros(L+1)
        dist = dist[dist != -1].flatten()
        distance_histogram += np.bincount(dist.astype(int), minlength=L+1)
        distance_histogram /= n_active_points / L**2
        total_histogram += distance_histogram

    total_histogram /= n_steps
    for i in range(L):
        if i < L//2:
            total_histogram[i] /= 4*i
        else:
            total_histogram[i] /= 4*(L-i)
    
    plt.plot(np.arange(L+1), total_histogram)
    plt.grid()
    plt.show()        




if __name__ == "__main__":
    main()
