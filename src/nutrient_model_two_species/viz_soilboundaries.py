import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def main():

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
    df['soil_empty_boundaries']  =df['soil_nonsoil_boundaries'] - df['soil_nutrient_boundaries'] - df['soil_worm_boundaries']
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
    critical_points = {0.2: 0.0144, 0.5: 0.0372, 1.0: 0.0414}

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

                axs[row, col].plot(theta_values, mean_values, marker='o', label=f'σ = {sigma}', color=color)
                axs[row, col].axvline(x=critical_points[sigma], color=color, linestyle='--', alpha=0.8)
            
            axs[row, col].set_title(f'{quantity.replace("_", " ").capitalize()}')
            axs[row, col].set_xlabel("Theta")
            axs[row, col].set_ylabel(f'Mean {quantity.replace("_", " ").capitalize()}')
            axs[row, col].grid()
            axs[row, col].legend()

    plt.tight_layout()
    plt.savefig('src/nutrient_model_two_species/plots/soil_boundaries/all_themes.png', dpi=300)
    plt.show()
    
if __name__ == "__main__":
    main()