# A single particle in the lattice

A 2D lattice is prepared filled with soil, and multiple 'bacteria' are seeded into it.

Periodic boundary conditions are applied.

In the animation:
- Brown = Soil
- White = Vacant space
- Black = Bacteria

## New dynamics (`single_species_RW_anim.py`)

Every time step:

* select a bacteria, and select an adjacent location
* if the adjacent location is another bacteria, do nothing
* if the adjacent location is space, move the bacteria
* if the adjacent location is soil, move the bacteria
* if the soil had at least one vacant neighbour before the move, attempt to reproduce into that neighbour with probability $r$
* randomly select $d$% of lattice sites, and if they're bacteria kill them off
* randomly select $s$% of lattice sites, and if they're vacant fill them with soil



## Old dynamics (`old_RW_anim.py`)

Every time step:

* select a bacteria, and move it to an adjacent location, leaving behind vacant space
* if the adjacent location is soil, the bacteria reproduces with a probability $r$ for each vacant space around it
* if the adjacent location is vacant space, the bacteria dies with a probability $d$
* vacant spaces are randomly filled with a probability $s$

I'd expect the soil should be relatively porus: there should be sparse trails of soil and vacancies caused by the bacteria.

**Note:** If the selected region is another bacteria, the bacteria 'consumes' the other one: it moves in, but two bacteria cannot occupy the same cell.

