## Meeting 1: 1st September, 2023

Two species of cellular automata: one which is big (2x2) and one which is (1x1)

Both move in a random walk

They move one step each time, and leave holes behind. Holes have a fixed probability to be filled

Big ones kill the small ones when they move in, and have a fixed death rate

Speciaes replicated when they have move from a hole to the soil, as resources are on the boundary.

Rate for death (when moves into a vacant space), rate for birth (when you move into birth), rate for hole filling

Calculate the density and fraction of soil-vacant walls.

Tuesday afternoon: Group meetings

Wednesday 10:30: Guest Lectures


## Meeting 2: 6th September, 2023

Do a plot of characteristics over time to look for oscillation.

Do it with random connections, and then you can use a mean-field comparison. Compare DEs to the solution.


## Meeting 3: 13th September, 2023

Logscale x and y

Asymmetry: bacteria ned to live longer than the holes

Check newrepro rate and reset simulation

**Every time step: move all bacteria**: Set a number of bacteria to move every time step, each can move multiple times. One movement per bacteria

Calculate cluster size distribution.

Do log-log axes on heatmap viz


## Meeting 4: 20th September, 2023

Try to make everything stochastic: choose a point, and depending on what it is, do something (fill/die/move)

If it's a bacteria, check for death, and if not make it move

Do phase space diagram for mean field model

Analyze meanfield soln by looking at bifurcation

The area where bacteria dies in mean-field and survives in neighbours could be relevant?

Do for 50x50, less phase points

Try to do it with lower birth rate

Make visualizations for every state, put them next to each other


## Meeting 5: 25th September, 2023

Visualize the nullcline: B-E landscape, change the parameters and see wher the fixed point

Try mean field

Try to quantify some sort of spatial structure which is causing the difference between mean field

Potentially try 3d.


## Meeting 6: 28th September, 2023

Pick a random side, you're going in an ordedred way.

Make  hybrid model, where you reprouce anywhere.

Find out how to vsualize 3d

Top triangle possible due to the sparsity of empty spaces.

Try to measure the spatial corrrelatials: the soil autocorrection.

Do emanfield for r=0.1


## Meeting 7: 4th October, 2023

One animal eats it's own environment, and thus it dies out faster than mean field?

CML: if density is coming in, reduce it by prob 1/d

Don't conserve d

Different types of vacancies, red bug can give the blue bug.

If red moves into blue, make it swapping

Slow repro rate, at a rate that would kill meanfield but locality stays alive

Try jumping far as well, and if you jump far, you have a chance to die


## Meeting 8: 11th October, 2023

Meanfield scaling r and d the same way doesn't change results

Try to do it with a larger lattice size and see if it approaches meanfield

Correct bug with red living and blue dying

Try nutrient, soil, empty, bacteria. Bacteria leave behind nutrients, and only replicate in them

If you move into soil, chance to replicate + chance to die

Try doing the soil filling only if your neighbours are soil


## Meeting 9: 18th October, 2023

Do mean field corresponding to soil neighbour

Check inside boundary of soil neighbour, is it still critical

Reduce replication rate and try

Add nullclines to presentation

Consider exploring nutrient model (to have bacteria support each other) and coupled map lattice

Do a bigger simulation and see if cutoff changes

See if exponent is independent of parameters s,d

Try to quantify scaling r,d

Search for predator prey model on lattice

Check power law in predator-prey

Do the meanfield for the predator prey


## Meeting 10: 25th October, 2023

When moving in, take over, or try to annhiliate

Low death rate: soil is limiting, high death rate: empty space is limiting.
Empty space is collapsing: bacteria deaths give empty sace, but not enough to compensate for empty space created in bacterial movement. No empty space causes bacteria to die out, which gives the phase transition.

Try to write the equations for the nullclines in both cases.

Try to do the 2 species with difference in death rate

Measure the boundary of cluster sizes. Fractical dimension of boundary

CML: measure autocorrelation without defining clusters

Think about using gradient instead of linear in CML, to emphasize boundaries


## Meeting 11: 1st November, 2023

Two species meanfield, and check fractions of red/blue

Gather more data for single species power law

Lotka voltairre with competitive exclusion

Microscopic biofilms(?)

Presentation notes:

- [x] Show predator prey vs meanfield

- [x] Add axes labels

- [x] Add a little discussion on power law

- [x] Remove the bad model

- [x] Remove the top heatmap

- [x] Explain predator-prey better


## Meeting 12: 3rd November 2023

Check if oscillations disappear with system size

calculate worm-worm correlation


## Meeting 13: 6th November 2023

Correlation function in space

Try reducing parameters by a facter and see if it changes stuff


## Meeting 14: 8th November 2023

Start from different survival rates:

Normalize the correlations

Try CML with new dynamics

Check cluster size dist for nutrient


## Meeting 15: 13th November 2023

Try changing the death rate and look for changes in the exponential for the worm-worm correlation

Check if changing the birth rate changes the worm-correlation

Check cluster size dist for different positions where it dies in meanfield

maybe check bistability in lattice model

Try adding nutrient to CML


## Meeting 16: 16th November 2023

So 200x200 and see if power law changes

Check fixed soil filling rate, vary death rate

Try  to replicate paper with 2-d

Try to make worms die from starvation: they die in white and live in soil

Check in what region the power law is seen

Check if power law changes with soil filling rate


## Meeting 17: 23rd November 2023

Worm -> Worm: kill the other worm

Small difference in the soil methodology

Try to calculate CSD for weird fractal dimension plot

## Meeting 18: 4th December 2023

At what point does the cutoff enter in 2d

Look for powerlaw regime with largest cluster size (1 order of magnitude less than system size)

Do the 3D raster

Try largest linear dimensions: both cumulative and in fractal dim


# Meeting 19: 7th December 2023

Normalize by the amount of soil in the system: cluster as % of total soil

CONNECT NUTRIENT AND WORM for clusters

lattice in worm oscillation amplitude (ratio in logscale)

Choose a soil site, and find the cluster size containing it over time. Move cluster along

Calculate effective soil filling rate

Show 3D over time (CSD)


# Meeting 20: 14th December 2023

Make 2D oscillations

Try smaller regions and see if the oscillations are visible

Max divided by min (or mean), to see the sensititity. Want to see how close it gets to "dying out"

Try the two species model


# Meeting 21: 21st December 2023

Try to keep a limit at 10^-6 in the meanfield

Try seeing if it comes back

2D, look at time series of the entire system, half system, quarter system, etc

Increase soil filling rate: multiple tries?

Try two species

Time distribution between soil disappearing and soil reappearing at a point


## Meeting 22: 28th December 2024

Look for oscillation in region where it survives in lattice but dies in mean-field

Check time period correlation between lattice and mean-field

Check how soil filling rate and death rate affect the exp decay rate

Check other lifetimes such as worm

Check meanfield death due to worm/empty/soil (in both regions, left and down, what's the cause?)

Check if oscillatory counts in 2spec

Check counts of simulations (see if coherent-oscillatory)

Try two spec with the same nutrient needed for both (maybe adding nutrient increases death rate)


## Meeting 23: 4th January 2024

Plot oscillation amplitude vs system size

Never let worm go below 10^-6

Check oscillations in 2d

Do well-mixed


## Meeting 24: 11th January 2024

Look up spiral simulation 

Try extreme case, 1 parasite, and 1 good worm

Look up other examples where oscillations => survival in lattice

Play around with tradeoff on death rate / nutrient production

Consider a distribution in parameters or allow parameters to mutate

Try parasite model in 2D

Population dynamics with delay DE : chaos


## Meeting 25: 18th January 2024

Check oscillation in yellow region

Check structure in yellow region

Can sustain multiple parasiets?

In parasite model, check lifetimes.

Run Lotka volterra for nutrient without nutrient to see difference for IUPAB

Play around with mu tradeoff being theta instead of rho


## Meeting 26: 24th January 2024

Check if power law at the top left point for CSD

Do more exhaustive check for oscillatory parasite, by checking lifetimes

Work on IUPAB abstract

Check 2D front or oscillations for large systems: maybe with rigid boundaries?

Lotka-Volterra: `soil_neighbour_rasterscan.html`, try to look for oscillations

Make the blocks smaller


## Meeting 27: 1st February 2024

Let blue swap places with soil, and see if oscillations retrun

Check correlation of diversity with power law

Read more about soil composition: empty fraction, living fraction, inorganic fraction

Try to generate a ton of differently-sized particles, from a power law dist of varying exp, and drop them and see the packing fraction


## Meeting 28: 8th February 2024

Try to do multiple worms in hogweg paper

Try to put a 3rd species in to eat the parasite

Check soil lifetimes calc

Check that blue dying doesn't give oscillations

Change mu1 to 1

Check if reducing deathrate of blue gives oscillations

Try to see if you can get a big clump + poweralw in nature

Look into dust powerlaw some more


## Meeting 29: 15th February 2024

Not much to discuss, showed Kuni the Hogweg plots


## Meeting 30: 19th February 2024

See if there's a long-term segregation in 2D hogweg

Dragon king state, chimera state (with decoherence)

Try 3D lattice visualization

Try different number of species

Try removing soil


## Meeting 31: 29th February 2024

Try moving into soil and replicating, and see if that changes localization

Try getting better params for localization

Check localization characteristic size

Check 2 clumps evolutions

Check cluster size of non-soil

Check if inside the cluster, you have soil as a power law

Try to do with parasite model localization

Try nutrient fast and worm slow

## "Meeting" 32: 4th-15th March 2024

(No kim, so spontaeneous meetings with Kuni)

Check for parasite localization

Check for powerlaw during parasite localization

Check whether LargeCluster + power law occurs more for parasite than single species

Potentially look at CML coarse-graining, with a spatial correlation for soil PSD?



