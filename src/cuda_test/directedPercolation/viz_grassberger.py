import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    p = 0.2873
    L = 1024
    df = pd.read_csv(f'src/cuda_test/directedPercolation/outputs/grassberger/p_{p}_L_{L}.csv')

    total_simulations = df['simulation'].nunique()

    # Calculate P(t)
    P_t = df.groupby('time')['simulation'].nunique() / total_simulations

    # Calculate n(t)
    all_times = pd.DataFrame({
        'time': np.repeat(df['time'].unique(), total_simulations),
        'simulation': np.tile(np.arange(total_simulations), len(df['time'].unique())),
    })
    df_full = pd.merge(all_times, df, how='left').fillna(0)
    n_t = df_full.groupby('time')['activeCounts'].mean()

    # Calculate R^2(t)
    R2_t = df.groupby('time')['R2'].mean()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    plt.suptitle(f'p={p}, L={L}, {total_simulations} simulations')

    # Plot P(t) on a log-log graph
    axs[0].plot(P_t.index, P_t, label='P(t)')
    axs[0].set_title('Survival probability')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('P(t)')
    axs[0].legend()
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].grid(True)
    ylim = axs[0].get_ylim()
    delta = -0.46
    x = np.array(P_t.index)
    axs[0].plot(x, x**delta, label=r'$\delta=$' + f'{delta} power law', linestyle='--', alpha=0.5)
    axs[0].legend()
    axs[0].set_ylim(ylim)

    # Plot n(t) on a log-log graph
    axs[1].plot(n_t.index, n_t, label='n(t)')
    axs[1].set_title('Number of active sites')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('n(t)')
    axs[1].legend()
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].grid(True)
    ylim = axs[1].get_ylim()
    eta = 0.214
    x = np.array(n_t.index)
    axs[1].plot(x, x**eta, label=r'$\eta=$' + f'{eta} power law', linestyle='--', alpha=0.5)
    axs[1].legend()
    axs[1].set_ylim(ylim)

    # Plot R^2(t) on a log-log graph
    axs[2].plot(R2_t.index, R2_t, label=r'R$^2$(t)')
    axs[2].set_title('Mean square radius of active sites')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel(r'R$^2$(t)')
    axs[2].legend()
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].grid(True)
    ylim = axs[2].get_ylim()
    z = 1.134
    x = np.array(R2_t.index)
    axs[2].plot(x, x**z, label=r'$z=$' + f'{z} power law', linestyle='--', alpha=0.5)
    axs[2].legend()
    axs[2].set_ylim(ylim)

    plt.tight_layout()
    # plt.savefig(f'src/cuda_test/directedPercolation/plots/grassberger/p_{p}_L_{L}.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()