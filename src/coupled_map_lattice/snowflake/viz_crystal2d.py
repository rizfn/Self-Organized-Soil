import matplotlib.pyplot as plt
import numpy as np

def main():
    Tc = 1
    T0 = -0.5
    L = 100
    D = 0.2
    C1 = 0.3
    C2 = 0.95
    filepath = f"src/coupled_map_lattice/snowflake/outputs/Tc_{Tc}_T0_{T0}_L_{L}_D_{D}_C1_{C1}_C2_{C2}.csv"
    x1 = np.loadtxt(filepath, delimiter=',')

    fig = plt.figure()
    cont = plt.contourf(np.arange(L), np.arange(L) ,x1,np.linspace(-0.5,1.5,num=10))
    contbar = plt.colorbar(cont)
    plt.xlabel('x [a.u.]')
    plt.ylabel('y [a.u.]')
    plt.title('CML model for crystal growth')
    plt.show()


if __name__ == "__main__":
    main()
