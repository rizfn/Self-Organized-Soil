import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('src/multi_species_nutrient/outputs/confinement/soilfracs_2D_parallel.csv')

N_values = df['N'].unique()
df.sort_values(by=['N', 'theta'], inplace=True)
plt.figure()

for N in N_values:
    subset = df[(df['N'] == N) & (df['mean_soil_fraction'] != 1)]
    if not subset.empty:
        plt.plot(subset['theta'], subset['mean_soil_fraction'], label=f'{N}spec', marker='x', linestyle='--')

plt.legend()
plt.xlabel('theta')
plt.ylabel('mean_soil_fraction')
plt.grid()
plt.show()