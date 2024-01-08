## Mean Field equations

$$
\begin{align}
\frac{\mathrm{d}S}{\mathrm{d}t} &= \sigma S (E + N) - S(W_G + W_B) \\
\frac{\mathrm{d}E}{\mathrm{d}t} &= S \big[ (1-\mu_1) W_G + (1-\mu_2) W_B \big]  + N \big[(1-\rho_1) W_G + (1-\rho_2) W_B \big]  + \theta \left(W_G+W_B\right) - \sigma S E \\
\frac{\mathrm{d}N}{\mathrm{d}t} &= S(\mu_1 W_G + \mu_2 W_G) - N(W_G + W_B) - \sigma S N \\
\frac{\mathrm{d}W_G}{\mathrm{d}t} &= \rho_1 W_G N - \theta W_G \\
\frac{\mathrm{d}W_B}{\mathrm{d}t} &= \rho_2 W_B N - \theta W_B
\end{align}
$$
