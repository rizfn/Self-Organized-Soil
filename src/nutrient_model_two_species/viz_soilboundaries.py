import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def main():
    L = 2048

    data_list = []

    for filename in glob(f"src/nutrient_model_two_species/outputs/sametheta/soil_boundaries/*.tsv"):
        sub_df = pd.read_csv(filename, sep='\t')
        sigma = float(filename.split("_")[5])
        theta = float(filename.split("_")[7].rsplit(".", 1)[0])
                
        sub_df["sigma"] = sigma
        sub_df["theta"] = theta

        data_list.append(sub_df)
    
    df = pd.concat(data_list)

    df['soil_nutrient_boundaries'] = df['soil_greennutrient_boundaries'] + df['soil_bluenutrient_boundaries']
    df['soil_worm_boundaries'] = df['soil_greenworm_boundaries'] + df['soil_blueworm_boundaries']
    df['soil_empty_boundaries'] = df['soil_nonsoil_boundaries'] - df['soil_nutrient_boundaries'] - df['soil_worm_boundaries']
    df['nutrients'] = df['green_nutrients'] + df['blue_nutrients']
    df['worms'] = df['green_worms'] + df['blue_worms']
    df['nutrient_production'] = df['green_nutrient_production'] + df['blue_nutrient_production']
    df['worm_production'] = df['green_worm_production'] + df['blue_worm_production']

    # Define themes
    themes = {
        "Soil Boundaries": ["soil_empty_boundaries", "soil_nutrient_boundaries", "soil_nonsoil_boundaries", "soil_worm_boundaries"],
        "Population Counts": ["emptys", "nutrients", "soil", "worms"],
        "Productions": ["empty_production", "nutrient_production", "soil_production", "worm_production"]
    }
    
    sigma_values = df["sigma"].unique()
    colors = plt.cm.Reds([0.5, 0.7, 1])
    critical_points = {0.3: 0.0315, 0.6: 0.039, 1.0: 0.04125}

    num_themes = len(themes)
    num_quantities_per_theme = max(len(quantities) for quantities in themes.values())
    fig, axs = plt.subplots(num_themes, num_quantities_per_theme, figsize=(5 * num_quantities_per_theme, 5 * num_themes), squeeze=False, sharex=True)

    for row, (theme, quantities) in enumerate(themes.items()):
        for col, quantity in enumerate(quantities):
            for sigma, color in zip(sigma_values, colors):
                same_sigma = df[df["sigma"] == sigma]
                theta_values = sorted(same_sigma["theta"].unique())
                mean_values = []

                for theta in theta_values:
                    same_theta = same_sigma[same_sigma["theta"] == theta]
                    mean_value = same_theta[quantity].mean()
                    mean_values.append(mean_value)

                axs[row, col].plot(theta_values, mean_values, marker='', label=f'σ = {sigma}', color=color)
                axs[row, col].axvline(x=critical_points[sigma], color=color, linestyle='--', alpha=0.8)
            
            axs[row, col].set_title(f'{quantity.replace("_", " ").capitalize()}')
            axs[row, col].set_xlabel("Theta")
            axs[row, col].set_ylabel(f'Mean {quantity.replace("_", " ").capitalize()}')
            axs[row, col].grid()
            axs[row, col].legend()

    plt.tight_layout()
    plt.savefig(f'src/nutrient_model_two_species/plots/soil_boundaries/all_themes_L_{L}.png', dpi=300)
    plt.show()


def plot_paper():
    L = 2048

    data_list = []

    for filename in glob(f"src/nutrient_model_two_species/outputs/sametheta/soil_boundaries/*.tsv"):
        # continue if file has  no lines
        if sum(1 for line in open(filename)) <= 1:
            continue
        sub_df = pd.read_csv(filename, sep='\t')
        sigma = float(filename.split("_")[5])
        theta = float(filename.split("_")[7].rsplit(".", 1)[0])
                
        sub_df["sigma"] = sigma
        sub_df["theta"] = theta

        data_list.append(sub_df)
    
    df = pd.concat(data_list)

    df['soil_nonsoil_boundaries'] /= (2 * L * L)
    df['nutrient_production'] = df['green_nutrient_production'] + df['blue_nutrient_production']

    quantities = ["soil_nonsoil_boundaries", "nutrient_production"]
    colors = ['#f48c06', '#dd1c1a', '#6a040f']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    plt.rcParams['font.size'] = 17

    fig, axs = plt.subplots(2, 1, figsize=(6.5, 12), sharex=True)
    axs = axs.flatten()

    sigma_values = df["sigma"].unique()
    sigma_values = [sigma for sigma in sigma_values if sigma != 0]

    for sigma, color in zip(sigma_values, colors):
        same_sigma = df[df["sigma"] == sigma]
        theta_values = sorted(same_sigma["theta"].unique())

        mean_values = {quantity: [] for quantity in quantities}

        for theta in theta_values:
            same_theta = same_sigma[same_sigma["theta"] == theta]
            mean_theta_values = same_theta[quantities].mean()

            for quantity in quantities:
                mean_values[quantity].append(mean_theta_values[quantity])

        for i, quantity in enumerate(quantities):
            axs[i].plot(theta_values, mean_values[quantity], marker='', color=color)

    # Get y_extent after plotting all lines
    y_extents = [axs[i].get_ylim()[1] - axs[i].get_ylim()[0] for i in range(len(quantities))]

    # critical_points = {0.3: 0.0315, 0.6: 0.039, 1.0: 0.04125}  # maybe a bit inaccurate?
    critical_points = {0.3: 0.031, 0.6: 0.0385, 1.0: 0.0411}
    for sigma, color in zip(sigma_values, colors):
        for i, quantity in enumerate(quantities):
            x_val = critical_points[sigma]
            same_sigma = df[df["sigma"] == sigma]
            theta_values = sorted(same_sigma["theta"].unique())
            mean_values = {quantity: [] for quantity in quantities}

            for theta in theta_values:
                same_theta = same_sigma[same_sigma["theta"] == theta]
                mean_value = same_theta[quantity].mean()
                mean_values[quantity].append(mean_value)

            if x_val < min(theta_values) or x_val > max(theta_values):
                continue
            left_idx = max(i for i in range(len(theta_values)) if theta_values[i] <= x_val)
            right_idx = min(i for i in range(len(theta_values)) if theta_values[i] >= x_val)
            if left_idx == right_idx:
                y_val = mean_values[quantity][left_idx]
            else:
                x_left, x_right = theta_values[left_idx], theta_values[right_idx]
                y_left, y_right = mean_values[quantity][left_idx], mean_values[quantity][right_idx]
                y_val = y_left + (y_right - y_left) * (x_val - x_left) / (x_right - x_left)
            arrow_height = 0.1 * y_extents[i]  # Set a fixed percentage of the y extent for the arrow height
            axs[i].annotate('', xy=(x_val, y_val), xytext=(x_val, y_val + arrow_height),
                            arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='-|>', lw=4, shrinkA=2, shrinkB=0))

    y_labels = ["Soil boundary fraction", "Nutrient production rate"]
    for i, quantity in enumerate(quantities):
        axs[i].set_ylabel(y_labels[i], fontsize=24)
        axs[i].grid()

    axs[-1].set_xlabel("Microbe death rate ($\\theta$)", fontsize=24)

    plt.tight_layout()
    plt.savefig('src/nutrient_model_two_species/plots/soil_boundaries/twospec_twonutrient_soilboundaries.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # main()
    plot_paper()