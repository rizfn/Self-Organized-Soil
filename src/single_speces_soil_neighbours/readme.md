# A single particle in the lattice

A 2D lattice is prepared filled with soil, and multiple 'bacteria' are seeded into it.

Periodic boundary conditions are applied.

In the animation:
- Brown = Soil
- White = Vacant space
- Black = Bacteria


## Stochastic dynamics (`soil_neighbour_raster.py`)

Every time step:

* choose a random site.
* if the site is soil, do nothing and advance to the next time step
* if the site is vacant, choose a random neighbour, and if they're soil try to fill it with probability $s$
* if the site is bacteria, check if it dies with probability $d$
  * if it's still alive, move it to a neigbouring site
  * if the neighbouring site was soil, choose a random neighbour that wasn't the original bacteria site
  * if the neighbour is vacant, attempt to replicate with probability $r$

**Note:** If a bacteria moves into another bacteria, nothing happens: they both remain bacteria.

The mean field ODEs can be written as

$$
\begin{align*}
\frac{\mathrm{d}S}{\mathrm{d}t} &= s \cdot E \cdot S - B \cdot S \\
\frac{\mathrm{d}E}{\mathrm{d}t} &= B \cdot S + d \cdot B - s \cdot E \cdot S - r \cdot B \cdot S \cdot E \\
\frac{\mathrm{d}B}{\mathrm{d}t} &= r \cdot B \cdot S \cdot E - d \cdot B
\end{align*}
$$


## Predator-Prey dynamics (`predator_prey_raster.py`)

The same as before, but now change the replication rules.

* If the chosen site is bacteria and wants to move to a neighbour which is soil, simply flip the soil to bacteria with rate $r$.

In other words, they move into soil and replicate behind them.

The mean field ODEs can be written as

$$
\begin{align*}
\frac{\mathrm{d}S}{\mathrm{d}t} &= s \cdot E \cdot S - r \cdot B \cdot S \\
\frac{\mathrm{d}E}{\mathrm{d}t} &= d \cdot B - s \cdot E \cdot S \\
\frac{\mathrm{d}B}{\mathrm{d}t} &= r \cdot B \cdot S - d \cdot B
\end{align*}
$$
