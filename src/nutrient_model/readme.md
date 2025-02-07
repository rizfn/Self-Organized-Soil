## Mean Field equations

$$
\begin{align}
\frac{\mathrm{d}S}{\mathrm{d}t} &= \sigma S (E + N) - W S \\
\frac{\mathrm{d}E}{\mathrm{d}t} &= (1-\rho) W N + \theta W - \sigma S E + \delta N \\
\frac{\mathrm{d}N}{\mathrm{d}t} &= W S - W N - \sigma S N - \delta N \\
\frac{\mathrm{d}W}{\mathrm{d}t} &= \rho W N - \theta W
\end{align}
$$

## Nullcline calculations:

### Equations of Nullclines

Simple case with no nutrient decay ($\delta=0$) and guaranteed reproduction ($\rho=1$):

Substituting $N=1-S-E-W$:

$$
\begin{align}
0 &= \sigma S (1-S-W) - W S \\
0 &= \theta W - \sigma S E \\
0 &= W (1-S-E-W) - \theta W
\end{align}
$$

First equation:
$$
\begin{align*}
    0 &= \sigma S (1-S-W) - W S \\
    0 &= S (\sigma-\sigma S-\sigma W - W) \\
    0 &= \sigma-\sigma S-\sigma W - W \\
    \sigma S &= \sigma - W(\sigma + 1) \\
    S &= \frac{\sigma - W(\sigma + 1)}{\sigma}
\end{align*}
$$

Second equation:
$$
\begin{align*}
    0 &= \theta W - \sigma S E \\
    \sigma S E  &= \theta W \\
    S &= \frac{\theta W}{\sigma E}
\end{align*}
$$

Third equation:
$$
\begin{align*}
    0 &= W (1-S-E-W) - \theta W \\
    0 &= W  (1-S-E-W-\theta) \\
    0 &= 1-S-E-W-\theta \\
    S &= 1-E-W-\theta
\end{align*}
$$

### Intersection of Nullclines

$S,E$ nullcline intersection:
$$
\begin{align*}
    W &= \frac{E\sigma}{E(\sigma + 1) + \theta} \\
    S &= \frac{\theta}{E(\sigma + 1) + \theta}
\end{align*}
$$

$S,W$ nullcline intersection:
$$
\begin{align*}
    W &= \sigma E + \sigma \theta \\
    S &= 1 - E(1 + \sigma) - \theta(1 + \sigma) 
\end{align*}
$$

$E,W$ nullcline intersection:
$$
\begin{align*}
    W &= \frac{\sigma E - \sigma E^2 - \sigma E \theta}{\theta + \sigma E} \\ 
    S &= 1 - E - \left(\frac{\sigma E - \sigma E^2 - \sigma E \theta}{\theta + \sigma E}\right) - \theta    
\end{align*}
$$

