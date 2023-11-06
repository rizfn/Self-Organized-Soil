import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def calc_correlation(soil_lattice, source_site=3, target_site=3):
    source_lattice = soil_lattice == source_site
    target_lattice = (soil_lattice == target_site).astype(np.int16)
    target_counts = np.roll(target_lattice, 1, axis=0) + np.roll(target_lattice, -1, axis=0) + np.roll(target_lattice, 1, axis=1) + np.roll(target_lattice, -1, axis=1)
    correlations = target_counts * source_lattice
    return correlations

def calc_correlation_factor(soil_lattice, source_site=3, target_site=3):
    correlation_lattice = calc_correlation(soil_lattice, source_site=source_site, target_site=target_site)
    observed_links = np.sum(correlation_lattice)
    number_sources = np.sum(soil_lattice == source_site)
    number_targets = np.sum(soil_lattice == target_site)
    L_sq = soil_lattice.shape[0]**2
    # number_nontargets = L_sq - number_targets
    # p_nbr_1 = 4 * number_targets / (L_sq) * (number_nontargets*(number_nontargets-1)*(number_nontargets-2) / (L_sq-1) / (L_sq-2) / (L_sq-3))
    # p_nbr_2 = 6 * number_targets * (number_targets-1) / (L_sq * (L_sq-1)) * (number_nontargets*(number_nontargets-1) / (L_sq-2) / (L_sq-3))
    # p_nbr_3 = 4 * number_targets * (number_targets-1) * (number_targets-2) / (L_sq * (L_sq-1) * (L_sq-2)) * (number_nontargets / (L_sq-3))
    # p_nbr_4 = number_targets * (number_targets-1) * (number_targets-2) * (number_targets-3) / (L_sq * (L_sq-1) * (L_sq-2) * (L_sq-3))
    # expected_links = number_sources * (1*p_nbr_1 + 2*p_nbr_2 + 3*p_nbr_3 + 4*p_nbr_4)
    expected_links = number_sources * 4 * number_targets / L_sq
    return observed_links / expected_links

def main():
    data = pd.read_json("docs/data/nutrient/lattice_rho=1_delta=0.json")

    source_site = 3  # 0 : empty, 1 : nutrient, 2 : soil, 3 : worm
    target_site = 3  # 0 : empty, 1 : nutrient, 2 : soil, 3 : worm

    data['correlations'] = data.soil_lattice.apply(lambda x: calc_correlation_factor(np.array(x), source_site, target_site))

    # Create a FacetGrid
    g = sns.FacetGrid(data, col="step", col_wrap=3)

    # Define a function to draw a heatmap on each facet
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index='sigma', columns='theta', values='correlations')
        ax = sns.heatmap(d, **kwargs)
        ax.invert_yaxis()
        xticks = [f'{tick:.1f}' for tick in ax.get_xticks()]
        yticks = [f'{tick:.1f}' for tick in ax.get_yticks()]
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)

    # Use the function on each facet
    g = g.map_dataframe(draw_heatmap, vmin=0, vmax=2, cmap="viridis", cbar=False)
    plt.colorbar(g.axes[-1].collections[0], ax=g.axes, orientation='vertical', fraction=.1)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
