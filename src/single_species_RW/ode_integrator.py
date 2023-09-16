import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def main():
    N = 20**2  # number of sites
    B_0 = int(N / 10)  # initial number of bacteria
    E_0 = int((N - B_0) / 2)  # initial number of empty sites
    S_0 = N - B_0 - E_0  # initial number of soil sites

    s = 0.02  # soil filling rate
    d = 0.02  # death rate
    r = 1  # reproduction rate

    stoptime = 10
    nsteps = 1_000_000
    dt = stoptime / nsteps

    S = [S_0]
    B = [B_0]
    E = [E_0]
    T = [0]

    for i in tqdm(range(nsteps)):
        S.append(S[i] + dt * (s*E[i] - B[i]*S[i]))
        E.append(E[i] + dt * (B[i]*S[i] + d*B[i] - s*E[i] - r*B[i]*S[i]*E[i]))
        B.append(B[i] + dt * (r*B[i]*S[i]*E[i] - d*B[i]))
        T.append(T[i] + dt)

    plt.plot(T, S, label="S")
    plt.plot(T, E, label="E")
    plt.plot(T, B, label="B")
    plt.plot(T, np.array(S) + np.array(E) + np.array(B) , label="S+E+B+T")
    plt.legend()
    # plt.yscale("log")
    plt.show()


    

if __name__ == "__main__":
    main()
