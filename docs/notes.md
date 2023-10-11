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


# Meeting 8: 11th October, 2023

Meanfield scaling r and d the same way doesn't change results

Try to do it with a larger lattice size and see if it approaches meanfield

Correct bug with red living and blue dying

Try nutrient, soil, empty, bacteria. Bacteria leave behind nutrients, and only replicate in them

If you move into soil, chance to replicate + chance to die

Try doing the soil filling only if your neighbours are soil
