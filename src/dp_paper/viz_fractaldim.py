import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import describe, Minuit
from scipy import stats

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    
def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])

class Chi2Regression:  # override the class with a better one
        
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        parameter_names = describe(self.f)[1:]
        self._parameters = {name: None for name in parameter_names}

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2

def set_matplotlib_params():
    colors = ['#901A1E', '#16BAC5', '#666666']
    plt.rcParams.update({'font.size': 16, 
                        #  'font.family': 'MS Reference Sans Serif', 
                         'axes.grid': True, 
                         'axes.prop_cycle': plt.cycler('color', colors),  # Set the color cycle
                         'grid.linestyle': '-',
                         'grid.linewidth': 0.5, 
                         'grid.alpha': 0.8, })


def main():

    set_matplotlib_params()

    p = 0.34375
    L = 4096
    df = pd.read_csv(f'src/dp_paper/outputs/FractalDim/p_{p}_L_{L}.tsv', sep='\t')

    # df = df[(df['box_length'] <= L/8) & (df['box_length'] >= 16)]

    N = df['box_length'].value_counts().values[0]  # Number of data points
    df = df.groupby('box_length').agg(['mean', 'std']).reset_index()  # get the mean and std of the box count

    def power_law(x, a, d):
        return a*x**d

    chi2_object = Chi2Regression(power_law, df['box_length'], df['box_count']['mean'], sy=df['box_count']['std']/np.sqrt(N))
    chi2_object.errordef = 1.0  # Chi2 definition (for Minuit)
    minuit = Minuit(chi2_object, a=df['box_count']['mean'][0], d=-1.89)  # What to minimize, what the starting values are
    minuit.migrad()  # Perform the minimisation
    Nvar = 2  # Number of variables (a, b, c, d)
    Ndof_fit = df['box_length'].nunique() - Nvar  # Number of degrees of freedom = Number of data points - Number of variables

    # Get the minimal value obtained for the quantity to be minimised (here the Chi2)
    Chi2_fit = minuit.fval                          # The chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)    # The chi2 probability given N degrees of freedom

    fig, ax = plt.subplots(figsize=(6, 4.5), tight_layout=True)
    plt.errorbar(df['box_length'], df['box_count']['mean'], yerr=df['box_count']['std']/np.sqrt(N), fmt='o', label='Data')
    x = np.linspace(df['box_length'].min(), df['box_length'].max(), 100)
    plt.plot(x, power_law(x, *minuit.values), label=f'$d_f$ = {-minuit.values["d"]:.3f} $\pm$ {minuit.errors["d"]:.3f}')
    plt.title(f'$\chi^2$={Chi2_fit:.2f}, p={Prob_fit:.2f}')
    plt.xlabel('Box length')
    plt.ylabel('Box count')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig(f'src/dp_paper/plots/fractalDim/p_{p}_L_{L}.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()