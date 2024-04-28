import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import glob

def main():

    filepath = 'src/simpleNbrGrowth/outputs/grassberger/'

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for file in glob.glob(filepath + '*.csv'):
        df = pd.read_csv(file)

        total_simulations = df['simulation'].nunique()

        # Calculate P(t), n(t), and R^2(t)
        P_t = df.groupby('time')['simulation'].nunique() / total_simulations
        n_t = df.groupby('time')['activeCounts'].mean()
        R2_t = df.groupby('time')['R2'].mean()

        # Extract sigma and theta from the filename using a regular expression
        match = re.search(r'sigma_(\d+\.?\d*)_theta_(\d+\.?\d*)\.csv$', file.split('\\')[-1])
        sigma = match.group(1)
        theta = match.group(2)

        # Plot P(t), n(t), and R^2(t) on the appropriate axes
        axs[0].plot(P_t.index, P_t, label=f'theta={theta}', alpha=0.8)
        axs[1].plot(n_t.index, n_t, label=f'theta={theta}', alpha=0.8)
        axs[2].plot(R2_t.index, R2_t, label=f'theta={theta}', alpha=0.8)

    plt.suptitle(f'$\sigma$={sigma}, {total_simulations} simulations')

    # Plot P(t) on a log-log graph
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
    axs[0].plot(x, x**delta, label=r'$\delta=$' + f'{delta} power law', linestyle='--')
    axs[0].legend()
    axs[0].set_ylim(ylim)

    # Plot n(t) on a log-log graph
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
    axs[1].plot(x, x**eta, label=r'$\eta=$' + f'{eta} power law', linestyle='--')
    axs[1].legend()
    axs[1].set_ylim(ylim)

    # Plot R^2(t) on a log-log graph
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
    axs[2].plot(x, x**z, label=r'$z=$' + f'{z} power law', linestyle='--')
    axs[2].legend()
    axs[2].set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(f'src/simpleNbrGrowth/plots/grassberger/sigma_{sigma}.png', dpi=300)
    plt.show()


def plot_one(sigma, theta):

    filepath = 'src/simpleNbrGrowth/outputs/grassberger/critical_point/'
    df_list = []

    for file in glob.glob(filepath + f'sigma_{sigma}_theta_{theta}_*.csv'):
        try:
            test_df = pd.read_csv(file)
            df_list.append(test_df)
        except pd.errors.EmptyDataError:
            continue
    
    df = pd.concat(df_list)

    total_simulations = df['simulation'].nunique()

    # Calculate P(t), n(t), and R^2(t)
    P_t = df.groupby('time')['simulation'].nunique() / total_simulations

    all_times = pd.DataFrame({
        'time': np.repeat(df['time'].unique(), total_simulations),
        'simulation': np.tile(np.arange(total_simulations), len(df['time'].unique())),
    })
    df_full = pd.merge(all_times, df, how='left').fillna(0)
    n_t = df_full.groupby('time')['activeCounts'].mean()

    R2_t = df.groupby('time')['R2'].mean()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle(f'$\sigma$={sigma}, {total_simulations} simulations')

    # Plot P(t), n(t), and R^2(t) on the appropriate axes
    axs[0].plot(P_t.index, P_t, label=f'theta={theta}', alpha=0.8)
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
    axs[0].plot(x, x**delta, label=r'$\delta=$' + f'{delta} power law', linestyle='--')
    axs[0].legend()
    axs[0].set_ylim(ylim)

    axs[1].plot(n_t.index, n_t, label=f'theta={theta}', alpha=0.8)
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
    axs[1].plot(x, x**eta, label=r'$\eta=$' + f'{eta} power law', linestyle='--')
    axs[1].legend()
    axs[1].set_ylim(ylim)

    axs[2].plot(R2_t.index, R2_t, label=f'theta={theta}', alpha=0.8)
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
    axs[2].plot(x, x**z, label=r'$z=$' + f'{z} power law', linestyle='--')
    axs[2].legend()
    axs[2].set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(f'src/simpleNbrGrowth/plots/grassberger/sigma_{sigma}_theta_{theta}.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    # main()
    plot_one(1, 0.6075)
    # plot_one(0.5, 0.305)