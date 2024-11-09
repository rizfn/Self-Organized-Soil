import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def main():
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Reds([0.5, 0.7, 1]))
    equilibrium_step_fraction = 3/4  # Fraction of the final steps to consider for the equilibrium
    data_list_2D = []

    for filename in glob("src/IUPAB_abstract/outputs/timeseriesWormCounts2D/*.csv"):
        sub_df = pd.read_csv(filename)
        sigma = filename.split("_")[2]
        theta = filename.split("_")[4].rsplit(".", 1)[0]
                
        sub_df["sigma"] = sigma
        sub_df["theta"] = theta
        data_list_2D.append(sub_df)
    
    df = pd.concat(data_list_2D)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for sigma in df["sigma"].unique():
        if sigma == "0":
            continue

        same_sigma = df[df["sigma"] == sigma]
        mean_worm_concentration = []
        mean_nutrient_concentration = []
        theta_values = []

        for theta in same_sigma["theta"].unique():
            if theta == "0":
                continue
            same_theta = same_sigma[same_sigma["theta"] == theta]
            final_fraction = same_theta.iloc[int(len(same_theta) * (1 - equilibrium_step_fraction)):]  # Select the final fraction steps
            mean_worm = final_fraction["greens"].mean()  # Calculate the mean worm concentration
            mean_nutrient = final_fraction["nutrients"].mean()  # Calculate the mean nutrient concentration
            mean_worm_concentration.append(mean_worm)
            mean_nutrient_concentration.append(mean_nutrient)
            theta_values.append(float(theta))

        ax1.plot(theta_values, mean_worm_concentration, marker='o', label=f'sigma = {sigma}')
        ax2.plot(theta_values, mean_nutrient_concentration, marker='o', label=f'sigma = {sigma}')

    ax1.set_title('Worm vs Theta')
    ax1.set_xlabel("Theta")
    ax1.set_ylabel("Mean Worm Concentration")
    ax1.grid()
    ax1.legend()

    ax2.set_title('Nutrient vs Theta')
    ax2.set_xlabel("Theta")
    ax2.set_ylabel("Mean Nutrient Concentration")
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.savefig('src/IUPAB_abstract/plots/worm_counts/3sigmas.png')
    plt.show()

if __name__ == "__main__":
    main()