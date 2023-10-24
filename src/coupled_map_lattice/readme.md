# Coupled map lattices

## Preliminary thoughts

Start with 2 lattices, one for soil densities, one for worm concentrations.

Then update iteratively, and hope to converge on some critical attractor.

### Possible dynamics:

* Smoothening of soil lattice: to take 'soil filling' into account. Each lattice side is influnced by it's neighbours, soil flows from high density to low density
* Worm birth/death: worms cannot survive in too high / too low density regions. Numbers rise if density is near 0.5, numbers lower if density is near 0 or 1. This could also be modelled as a 'reshuffling' in order to keep the worm field conservative
* Worm action on soil: High worm counts would push soil away to neigbours, thus affecting the densities.

### Implementation `conserved_soil.py`:

Soil lattice is conserved, but worm lattice isn't. Soil can only move, never appear or disappear.

* **Soil lattice smoothening:** Each site goes through a mean filter kernel, with a `smoothening_factor`.
* **Worm birth/death:** Target worm count goes up linearly from 0 to 1 as density increases to 0.5, then down as density increases. Approach target worm count with a `birth_factor`
* **Worm action on soil:** High worm counts would push soil away to neigbours, thus affecting the densities. The fraction of soil pushed depends on an `interaction_factor`


