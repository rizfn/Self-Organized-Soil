# A single particle in the lattice

A 2D lattice is prepared filled with soil, and multiple 'bacteria' are seeded into it.

Every time step:

* select a bacteria, and move it to an adjacent location, leaving behind vacant space
* if the adjacent location is soil, the bacteria reproduces with a probability $r$ for each vacant space around it
* if the adjacent location is vacant space, the bacteria dies with a probability $d$
* vacant spaces are randomly filled with a probability $s$

I'd expect the soil should be relatively porus: there should be sparse trails of soil and vacancies caused by the bacteria.

**Note:** If the selected region is another bacteria, the bacteria 'consumes' the other one: it moves in, but two bacteria cannot occupy the same cell.

Periodic boundary conditions are applied.

In the animation:
- Brown = Soil
- White = Vacant space
- Black = Bacteria