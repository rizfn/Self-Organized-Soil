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

if __name__ == "__main__":
    main()