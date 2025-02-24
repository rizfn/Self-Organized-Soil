import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def plot_all():
    # Define the colors for the different sigma values
    colors = plt.cm.Reds([0.5, 0.7, 1])
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    equilibrium_step_fraction = 3/4  # Fraction of the final steps to consider for the equilibrium
    data_list_2D = []

    for filename in glob("src/IUPAB_abstract/outputs/soil_boundaries/*.tsv"):
        sub_df = pd.read_csv(filename, sep='\t')
        sigma = filename.split("_")[3]
        theta = filename.split("_")[5].rsplit(".", 1)[0]
                
        sub_df["sigma"] = sigma
        sub_df["theta"] = theta
        data_list_2D.append(sub_df)
    
    df = pd.concat(data_list_2D)

    quantities = ["soil_nonsoil_boundaries", "soil_worm_boundaries", "soil_nutrient_boundaries",
                  "emptys", "nutrients", "soil", "worms",
                  "empty_production", "nutrient_production", "soil_production", "worm_production"]
    num_quantities = len(quantities)
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    axs = axs.flatten()

    sigma_values = df["sigma"].unique()
    sigma_values = [sigma for sigma in sigma_values if sigma != "0"]

    for sigma, color in zip(sigma_values, colors):
        same_sigma = df[df["sigma"] == sigma]
        theta_values = []

        mean_values = {quantity: [] for quantity in quantities}

        for theta in same_sigma["theta"].unique():
            if theta == "0":
                continue
            same_theta = same_sigma[same_sigma["theta"] == theta]
            final_fraction = same_theta.iloc[int(len(same_theta) * (1 - equilibrium_step_fraction)):]  # Select the final fraction steps
            theta_values.append(float(theta))

            for quantity in quantities:
                mean_value = final_fraction[quantity].mean()  # Calculate the mean value for the quantity
                mean_values[quantity].append(mean_value)

        for i, quantity in enumerate(quantities):
            axs[i].plot(theta_values, mean_values[quantity], marker='o', label=f'sigma = {sigma}', color=color)

    for i, quantity in enumerate(quantities):
        axs[i].set_title(f'{quantity.capitalize()} vs Theta')
        axs[i].set_xlabel("Theta")
        axs[i].set_ylabel(f'Mean {quantity.capitalize()}')
        axs[i].grid()
        axs[i].legend()

        # Add vertical dashed lines
        axs[i].axvline(x=0.13, color=colors[0], linestyle='--', label='sigma=0.1 soil power law', alpha=0.8)
        axs[i].axvline(x=0.14, color=colors[1], linestyle='--', label='sigma=0.5 soil power law', alpha=0.8)
        axs[i].axvline(x=0.134, color=colors[2], linestyle='--', label='sigma=1.0 soil power law', alpha=0.8)

    plt.tight_layout()
    plt.savefig('src/IUPAB_abstract/plots/worm_counts/3sigmas_soilboundaries_2D.png')
    plt.show()



def main():
    # Define the colors for the different sigma values
    colors = ['#f48c06', '#dd1c1a', '#6a040f']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    plt.rcParams['font.size'] = 16
    equilibrium_step_fraction = 3/4  # Fraction of the final steps to consider for the equilibrium
    data_list_2D = []

    for filename in glob("src/IUPAB_abstract/outputs/soil_boundaries/*.tsv"):
        sub_df = pd.read_csv(filename, sep='\t')
        sigma = filename.split("_")[3]
        theta = filename.split("_")[5].rsplit(".", 1)[0]
                
        sub_df["sigma"] = sigma
        sub_df["theta"] = theta
        data_list_2D.append(sub_df)
    
    df = pd.concat(data_list_2D)

    quantities = ["soil_nonsoil_boundaries", "nutrient_production"]
    num_quantities = len(quantities)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.flatten()

    sigma_values = df["sigma"].unique()
    sigma_values = [sigma for sigma in sigma_values if sigma != "0"]

    for sigma, color in zip(sigma_values, colors):
        same_sigma = df[df["sigma"] == sigma]
        theta_values = []

        mean_values = {quantity: [] for quantity in quantities}

        for theta in same_sigma["theta"].unique():
            if theta == "0":
                continue
            same_theta = same_sigma[same_sigma["theta"] == theta]
            final_fraction = same_theta.iloc[int(len(same_theta) * (1 - equilibrium_step_fraction)):]  # Select the final fraction steps
            theta_values.append(float(theta))

            for quantity in quantities:
                mean_value = final_fraction[quantity].mean()  # Calculate the mean value for the quantity
                mean_values[quantity].append(mean_value)

        for i, quantity in enumerate(quantities):
            axs[i].plot(theta_values, mean_values[quantity], marker='', label=f'$\sigma$ = {sigma}', color=color)

    # Get y_extent after plotting all lines
    y_extents = [axs[i].get_ylim()[1] - axs[i].get_ylim()[0] for i in range(num_quantities)]

    for sigma, color in zip(sigma_values, colors):
        same_sigma = df[df["sigma"] == sigma]
        theta_values = []

        mean_values = {quantity: [] for quantity in quantities}

        for theta in same_sigma["theta"].unique():
            if theta == "0":
                continue
            same_theta = same_sigma[same_sigma["theta"] == theta]
            final_fraction = same_theta.iloc[int(len(same_theta) * (1 - equilibrium_step_fraction)):]  # Select the final fraction steps
            theta_values.append(float(theta))

            for quantity in quantities:
                mean_value = final_fraction[quantity].mean()  # Calculate the mean value for the quantity
                mean_values[quantity].append(mean_value)

        for i, quantity in enumerate(quantities):
            # Add arrows instead of vertical dashed lines
            # x_val_map = {0.1: 0.13, 0.5: 0.14, 1.0: 0.134}  # maybe a bit unaccurate?
            x_val_map = {0.1: 0.128, 0.5: 0.138, 1.0: 0.132}
            if float(sigma) in x_val_map:
                x_val = x_val_map[float(sigma)]
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
        axs[i].set_xlabel("Microbe death rate ($\\theta$)", fontsize=20)
        axs[i].set_ylabel(y_labels[i], fontsize=20)
        axs[i].grid()

    plt.tight_layout()
    plt.savefig('src/IUPAB_abstract/plots/worm_counts/2quantities_soilboundaries_2D.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    main()
    # plot_all()
