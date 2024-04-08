import matplotlib.pyplot as plt
import glob
import numpy as np

# Get all file names
files = glob.glob('src/cuda_test/directedPercolation/outputs/timeseries1D/*.tsv')

# Sort files into three categories
directed_percolation_files = sorted([file for file in files if 'dirP' in file.rsplit('/', 1)[-1]])
model_files = sorted([file for file in files if 'theta_' in file.rsplit('/', 1)[-1]])
overwrite_files = sorted([file for file in files if 'overwrite' in file.rsplit('/', 1)[-1]])

# Create subplots
fig, axs = plt.subplots(4, 3, figsize=(20, 15))

# Plot overwrite files
for i, file in enumerate(overwrite_files[:4]):
    data = np.genfromtxt(file, delimiter=',', skip_footer=1)
    axs[i, 0].imshow(data.T, aspect='auto', cmap='binary')
    axs[i, 0].set_title(file.rsplit('/', 1)[-1].rsplit('\\', 1)[-1])

# Plot directed percolation files
for i, file in enumerate(directed_percolation_files[:4]):
    data = np.genfromtxt(file, delimiter=',', skip_footer=1)
    axs[i, 1].imshow(data.T, aspect='auto', cmap='binary')
    axs[i, 1].set_title(file.rsplit('/', 1)[-1].rsplit('\\', 1)[-1])

# Plot model files
for i, file in enumerate(model_files[:4]):
    data = np.genfromtxt(file, delimiter=',', skip_footer=1)
    axs[i, 2].imshow(data.T, aspect='auto', cmap='binary')
    axs[i, 2].set_title(file.rsplit('/', 1)[-1].rsplit('\\', 1)[-1])

plt.tight_layout()
plt.savefig('src/cuda_test/directedPercolation/plots/directed_percolation.png')
plt.show()