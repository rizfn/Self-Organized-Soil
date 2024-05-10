import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay

def main():
    N = 10
    np.random.seed(154)
    points = np.random.rand(N, 2)
    vor = Voronoi(points)
    delaunay = Delaunay(points)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#901A1E'])
    fig, axs = plt.subplots(1, 3, figsize=(27, 6))
    voronoi_plot_2d(vor, show_vertices=False, show_points=True, line_colors='#666666', line_width=2, line_alpha=1, point_size=10, ax=axs[0])  
    axs[0].set_axis_off()  
    # Plot Delaunay triangulation
    axs[1].triplot(points[:, 0], points[:, 1], delaunay.simplices.copy(), color='#666666')
    axs[1].plot(points[:, 0], points[:, 1], 'o', color='#901A1E')
    axs[1].set_axis_off()
    # For third axis, plot both Delaunay and Voronoi
    voronoi_plot_2d(vor, show_vertices=False, show_points=True, line_colors='#666666', line_width=2, line_alpha=1, point_size=10, ax=axs[2])
    axs[2].triplot(points[:, 0], points[:, 1], delaunay.simplices.copy(), color='#901A1E', linestyle='-.')
    axs[2].set_axis_off()

    plt.savefig('src/visualizations_simple/plots/voronoi_delaunay.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

if __name__ == "__main__":
    main()
