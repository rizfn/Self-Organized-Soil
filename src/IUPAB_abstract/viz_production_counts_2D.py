import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def main():
    # Define the colors for the different sigma values
    colors = plt.cm.Reds([0.5, 0.7, 1])
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    equilibrium_step_fraction = 3/4  # Fraction of the final steps to consider for the equilibrium
    data_list_2D = []

    for filename in glob("src/IUPAB_abstract/outputs/timeseriesWormCounts2DProd/*.csv"):
        sub_df = pd.read_csv(filename)
        sigma = filename.split("_")[2]
        theta = filename.split("_")[4].rsplit(".", 1)[0]
                
        sub_df["sigma"] = sigma
        sub_df["theta"] = theta
        data_list_2D.append(sub_df)
    
    df = pd.concat(data_list_2D)

    quantities = ["emptys", "nutrients", "soil", "greens", "empty_production", "nutrient_production", "soil_production", "worm_production"]
    num_quantities = len(quantities)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
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
    plt.savefig('src/IUPAB_abstract/plots/worm_counts/3sigmas_2D_allcounts.png')
    plt.show()

if __name__ == "__main__":
    main()