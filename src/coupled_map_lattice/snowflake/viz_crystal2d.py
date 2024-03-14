import matplotlib.pyplot as plt
import numpy as np

def main():
    Tc = 1
    T0 = -0.6
    L = 500
    D = 0.2
    C1 = 0.075
    C2 = 0.25
    filepath = f"src/coupled_map_lattice/snowflake/outputs/Tc_{Tc}_T0_{T0}_L_{L}_D_{D}_C1_{C1}_C2_{C2}.csv"
    # filepath = f"src/coupled_map_lattice/snowflake/outputs/Tc_0_T0_-0.2_L_{L}_D_0.15_C_1.csv"
    x1 = np.loadtxt(filepath, delimiter=',')

    # cont = plt.contourf(np.arange(L), np.arange(L), x1, np.linspace(-0.5,1.5,num=10))
    cont = plt.imshow(x1)

    contbar = plt.colorbar(cont)
    plt.xlabel('x [a.u.]')
    plt.ylabel('y [a.u.]')
    plt.title(f'$T_c$={Tc}, $T_0$={T0}, $L$={L}, $D$={D}, $C_1$={C1}, $C_2$={C2}')
    plt.savefig(f"src/coupled_map_lattice/snowflake/plots/snowflake_test/Tc_{Tc}_T0_{T0}_L_{L}_D_{D}_C1_{C1}_C2_{C2}.png")
    plt.show()


if __name__ == "__main__":
    main()
