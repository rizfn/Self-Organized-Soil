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
* if the site is vacant, try to fill it with probability $s$
* if the site is bacteria, check if it dies with probability $d$
  * if it's still alive, move it to a neigbouring site
  * if the neighbouring site was soil, choose a random neighbour that wasn't the original bacteria site
  * if the neighbour is vacant, attempt to replicate with probability $r$

**Note:** If a bacteria moves into another bacteria, nothing happens: they both remain bacteria.


## "Parallel" dynamics

Every time step:

* select all empty sites, and fill each one with a probability $s$
* select all bacteria sites, and kill each one with a probability $d$
* Perform one 'move' for every bacteria in the system. A move consists of:
  * select a bacteria, and select an adjacent location
  * if the adjacent location is another bacteria, do nothing
  * if the adjacent location is space, move the bacteria
  * if the adjacent location is soil, move the bacteria
  * if the soil had at least one vacant neighbour before the move, attempt to reproduce into that neighbour with probability $r$


## Old dynamics (`old_RW_anim.py`)

Every time step:

* select a single bacteria, and move it to an adjacent location, leaving behind vacant space
* if the adjacent location is soil, the bacteria reproduces with a probability $r$ for each vacant space around it
* if the adjacent location is vacant space, the bacteria dies with a probability $d$
* vacant spaces are randomly filled with a probability $s$

I'd expect the soil should be relatively porus: there should be sparse trails of soil and vacancies caused by the bacteria.

**Note:** If the selected region is another bacteria, the bacteria 'consumes' the other one: it moves in, but two bacteria cannot occupy the same cell.

