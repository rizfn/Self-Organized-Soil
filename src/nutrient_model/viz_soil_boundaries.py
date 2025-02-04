import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def main():
    data_list = []

    for filename in glob("src/nutrient_model/outputs/soil_boundaries/*.tsv"):
        sub_df = pd.read_csv(filename, sep='\t')
        sigma = filename.split("_")[3]
        theta = float(filename.split("_")[5].rsplit(".", 1)[0])  # Convert theta to float
                
        sub_df["sigma"] = sigma
        sub_df["theta"] = theta
        data_list.append(sub_df)
    
    df = pd.concat(data_list)

    quantities = ["soil_nonsoil_boundaries", "soil_worm_boundaries", "soil_nutrient_boundaries"]
    sigma_values = df["sigma"].unique()
    num_sigmas = len(sigma_values)
    fig, axs = plt.subplots(num_sigmas, 3, figsize=(20, 5 * num_sigmas))
    axs = axs.reshape((num_sigmas, 3))

    for row, sigma in enumerate(sigma_values):
        same_sigma = df[df["sigma"] == sigma]
        theta_values = sorted(same_sigma["theta"].unique())  # Sort theta values as floats

        mean_values = {quantity: [] for quantity in quantities}

        for theta in theta_values:
            same_theta = same_sigma[same_sigma["theta"] == theta]
            mean_theta_values = same_theta[quantities].mean()

            for quantity in quantities:
                mean_values[quantity].append(mean_theta_values[quantity])

        for col, quantity in enumerate(quantities):
            axs[row, col].plot(theta_values, mean_values[quantity], marker='o', label=f'sigma = {sigma}')
            axs[row, col].set_title(f'{quantity.replace("_", " ").capitalize()}')
            axs[row, col].set_xlabel("Theta")
            axs[row, col].set_ylabel(f'Mean {quantity.replace("_", " ").capitalize()}')
            axs[row, col].grid()
            axs[row, col].legend()

            # Add vertical dashed lines based on sigma value
            if sigma == "0.1":
                axs[row, col].axvline(x=0.13, linestyle='--', label='sigma=0.1 soil power law', alpha=0.8)
            elif sigma == "0.5":
                axs[row, col].axvline(x=0.14, linestyle='--', label='sigma=0.5 soil power law', alpha=0.8)
            elif sigma == "1":
                axs[row, col].axvline(x=0.134, linestyle='--', label='sigma=1.0 soil power law', alpha=0.8)

    plt.tight_layout()
    plt.savefig("src/nutrient_model/plots/soil_boundaries/2D.png")
    plt.show()


def plot_paper():
    # Define the colors for the different sigma values
    colors = plt.cm.Reds([0.5, 0.7, 1])
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    equilibrium_step_fraction = 3/4  # Fraction of the final steps to consider for the equilibrium
    data_list_2D = []

    for filename in glob("src/nutrient_model/outputs/soil_boundaries/*.tsv"):
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
            axs[i].plot(theta_values, mean_values[quantity], marker='o', label=f'sigma = {sigma}', color=color)

    y_labels = ["Soil boundary fraction", "Nutrient production"]
    for i, quantity in enumerate(quantities):
        axs[i].set_xlabel("Theta")
        axs[i].set_ylabel(y_labels[i])
        axs[i].grid()
        axs[i].legend()

        # Add vertical dashed lines
        axs[i].axvline(x=0.13, color=colors[0], linestyle='--', label='sigma=0.1 soil power law', alpha=0.8)
        axs[i].axvline(x=0.14, color=colors[1], linestyle='--', label='sigma=0.5 soil power law', alpha=0.8)
        axs[i].axvline(x=0.134, color=colors[2], linestyle='--', label='sigma=1.0 soil power law', alpha=0.8)

    plt.tight_layout()
    plt.savefig('src/nutrient_model/plots/soil_boundaries/2quantities_soilboundaries_2D.png')
    plt.show()

if __name__ == "__main__":
    # main()
    plot_paper()