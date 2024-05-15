import numpy as np
import csv
import sys


def load_csv(filename):
    maxInt = sys.maxsize
    decrement = True

    while decrement:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.
        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt/10)
            decrement = True

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip the header
        steps = []
        filled_counts = []
        empty_counts = []
        for row in reader:
            steps.append(int(row[0]))  # Convert to int and add to steps
            # Check if the row is empty before splitting and converting to int
            filled_counts.append(sum(int(x) for x in row[1].split(',') if x) if row[1] else 0)
            empty_counts.append(sum(int(x) for x in row[2].split(',') if x) if row[2] else 0)
    
    return np.array(steps), np.array(filled_counts), np.array(empty_counts)


def main(filename, is_filled):
    steps, filled_counts, empty_counts = load_csv(filename)
    total_counts = filled_counts + empty_counts
    if is_filled:
        densities = filled_counts / total_counts
    else:
        densities = empty_counts / total_counts
    mean_density = np.mean(densities)
    std_density = np.std(densities)
    error_density = std_density / np.sqrt(len(densities))  # Standard error of the mean
    print("Mean density of lattice: ", mean_density, "±", error_density)

if __name__ == '__main__':
    # main('src/directedPercolation/outputs/CSD2D/criticalPointsLarge/p_0.344_L_4096.tsv', 1)  # 0.5768964433670044 ± 1.7593527413279654e-05
    # main('src/directedPercolation/outputs/CSD2D/criticalPointsLarge/p_0.318_L_4096.tsv', 0)  # 0.5729842209815978 ± 1.907862655398344e-05
    main('src/directedPercolation/outputs/CSD2D/criticalPoints/p_0.318_L_1024.tsv', 0) # super 0.7863325834274292 ± 4.4954861377560076e-05
    main('src/directedPercolation/outputs/CSD2D/criticalPoints/p_0.3185_L_1024.tsv', 0) # crit 0.5695032978057861 ± 7.842277239289493e-05
    main('src/directedPercolation/outputs/CSD2D/criticalPoints/p_0.319_L_1024.tsv', 0) # sub 0.565850715637207 ± 8.850698023196516e-05
