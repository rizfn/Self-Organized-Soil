# A single particle in the lattice

A 2D lattice is prepared filled with soil, and multiple 'bacteria' are seeded into it.

Periodic boundary conditions are applied.

In the animation:
- Brown = Soil
- White = Vacant space
- Black = Bacteria


## Stochastic dynamics (`single_species_RW_anim.py`)

Every time step:

* choose a random site.
* if the site is soil, do nothing and advance to the next time step
* if the site is vacant, choose a random neighbour, and if they're soil try to fill it with probability $s$
* if the site is bacteria, check if it dies with probability $d$
  * if it's still alive, move it to a neigbouring site
  * if the neighbouring site was soil, choose a random neighbour that wasn't the original bacteria site
  * if the neighbour is vacant, attempt to replicate with probability $r$

**Note:** If a bacteria moves into another bacteria, nothing happens: they both remain bacteria.

