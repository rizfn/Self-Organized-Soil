---
marp: true
theme: uncover
math: mathjax
paginate: true
_paginate: skip
# backgroundImage: url('images/defence/frontpage/ku-outline-seal.png')
backgroundImage: url('images/PhD/KU5.png')
backgroundSize: contain
style: |
        .columns {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.6rem;
        }
        h1, h2, h3, h4, h5, h6, strong {
          color: #400000;
        }
        .caption {
          font-size: 0.5em;
          line-height: 10%;
          letter-spacing: 0.01em;
          margin-top: -100px;
          margin-bottom: -100px;
        }

---


![bg right:42% brightness:1.5 saturate:1.8](images/soil.png)

# Self-Organised Soil

$\\$
Riz Fernando Noronha

Supervised by Kim Sneppen and Kunihiko Kaneko


---

## Why soil?
$\\$

- Soil drives agriculture, and helps sustain the biosphere
<br>
- Ecosystem for organisms across the tree of life
<br>
- Extremely important for humanity


---

<div class='columns'>


- Soil particle-sizes follow a power-law

- Typically explained through *fragmentation*: Soil $\approx$ fragmented rocks

- Fragmentation follows a power law, and can be used to explain soil

<img src="images/feb29/tyler_wheatcraft.png" style="max-width: 90%;"></img>
<span class="caption"><a href="https://acsess.onlinelibrary.wiley.com/doi/abs/10.2136/sssaj1989.03615995005300040001x">Tyler and Wheatcraft, 1989</a></span>

</div>

---

![bg fit right:53.5%](images/defence/soil_species_percentages.png)

- Soil has a lot of life!
<br>
- Modelling as a purely inorganic substance is not enough
<br>
- *'Biomaterial'*
<br>

<span class="caption"><a href="https://www.pnas.org/doi/full/10.1073/pnas.2304663120">Anthony, Bender, & Heijden, 2023</a></span>

---

### Biological impact on soil structure

- Three types of soil:
Control, Bulk, Rhizophere

- Biota increases from left to right

![width:800](images/defence/feeney_microbes_soilpores.png)
<span class="caption"><a href="https://www.pnas.org/doi/full/10.1073/pnas.2304663120">Feeney et al, 2006</a></span>


---

- **Soil porosity** appears to increase

- *Larger particles* (>2μm) of soil are observed

- Perhaps biologically induced **soil-aggregation**?


<div class="columns">

<img src="images/defence/feeney_porosity.png" style="max-width: 100%;"></img>

<img src="images/defence/feeney_soil_cluster_size.png" style="max-width: 100%;"></img>

</div>

<span class="caption"><a href="https://www.pnas.org/doi/full/10.1073/pnas.2304663120">Feeney et al 2006</a></span>

---


### Parameters

- $\sigma$: Soil filling rate
- $\theta$: Death rate of worms
- $\mu$: Nutrient generation rate $=1$
- $\rho$: Reproductive rate of worms $=1$

$\\$

> *One is a great number!*

---

### Algorithm

<div class="columns">

<img src="images/feb29/algo_network.png" style="max-width: 100%; "></img>

- Worms convert soil to nutrients
- Worms use nutrients to reproduce
- Soil 'grows' from empty and nutrients
- Predator-prey with time-delay

</div>

---

### Mean-Field Equations

$$
\definecolor{darkergrey}{rgb}{0.35, 0.35, 0.35}
\begin{align}
\frac{\mathrm{d}\textcolor{brown}{S}}{\mathrm{d}t} &= \sigma \textcolor{brown}{S} (\textcolor{darkergrey}{E} + \textcolor{teal}{N}) - \textcolor{green}{W} \textcolor{brown}{S} \\
\frac{\mathrm{d}\textcolor{darkergrey}{E}}{\mathrm{d}t} &= (1-\rho) \textcolor{green}{W} \textcolor{teal}{N} + (1-\mu) \textcolor{green}{W} \textcolor{brown}{S} + \theta \textcolor{green}{W} - \sigma \textcolor{brown}{S} \textcolor{darkergrey}{E} \\
\frac{\mathrm{d}\textcolor{teal}{N}}{\mathrm{d}t} &= \mu \textcolor{green}{W} \textcolor{brown}{S} - \textcolor{green}{W} \textcolor{teal}{N} - \sigma \textcolor{brown}{S} \textcolor{teal}{N} \\
\frac{\mathrm{d}\textcolor{green}{W}}{\mathrm{d}t} &= \rho \textcolor{green}{W} \textcolor{teal}{N} - \theta \textcolor{green}{W}
\end{align}
$$

---

<iframe width="auto" height="1000px" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/nutrient_meanfield_attractors.html" style="border: 1px solid #cccccc" frameborder=0>
</iframe>


---

### Cellular Automaton Model

$\\$

<div class="columns">

- Same dynamics, on a lattice

- Worm create nutrients *in their vicinity*

- Worms thus help each other survive

<img src="images/feb29/algo_grid.png" style="max-width: 100%; "></img>

</div>

---

<iframe width="auto" height="1000px" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/defence/nutrient_lattice_density.html" style="border: 1px solid #ccc" frameborder=0>
</iframe>

---

<iframe width="auto" height="1000px" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/defence/nutrient_lattice_3D_density.html" style="border: 1px solid #ccc" frameborder=0>
</iframe>

---

### Survival in lattice vs Meanfield

![w:1150](images/defence/rasterscan_attractors.png)

---

Spatial structure **damps** oscillations

![w:1100](images/defence/nutrient_timeseries_raster_snic.png)


---

### Two Species

$$\\$$

- Add a second species, and make it asymmetric in terms of:
  - $\mu$, nutrient generation rate
  - $\rho$, reproduction rate
<br>
- $\mu=0$ is a *parasite*.

---

#### Mean-Field: Parasite Problem

$\\$

![w:800px](images/feb29/competitive_exclusion.png)

- Worms don't care who creates the nutrient
- A slight advantage mean you dominate
- **Coexistence is impossible!**
  *Competitive exclusion*: Higher $\rho$ always wins


---

<iframe width="auto" height="1000px" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/defence/twospec_samenutrient_lattice_st_density.html" style="border: 1px solid #ccc" frameborder=0>
</iframe>


---

### Soil particle sizes

![width:800](images/defence/soil_cluster_size36.png)

Nearest neighbour connected components for clusters:
$\implies$ both are a cluster of size 36!

---

### Cluster size distributions

![width:1100](images/defence/csd_sub_crit_super.png)


- For certain parameter values, we do see power-law distributed "particle sizes"

- Exponent $\approx 1.85$ (in 2D)


---

## Directed Percolation

<div class="columns">

<img src="images/defence/directed_percolation.png" style="max-width: 100%; "></img>

- One 'preferred' direction (time)

- Each site 'spreads' into neighbouring sites

- Three critical exponents
  - $\rho^\text{stat} \sim (p-p_c)^\beta$
  - $\xi_\parallel \ \ \ \, \sim (p - p_c)^{-\nu_\parallel}$
  - $\xi_\perp \ \ \ \sim (p - p_c)^{-\nu_\perp}$

</div>


---

![bg fit right:30%](images/defence/DP_1D_slice.png)

### Investigating Cluster Size Distributions


- Take a slice at *constant time*

- Observe the distribution of filled and empty clusters

- 1+1D: Empty CSD = Distance between branches ($p$=$p_c$)

- Filled CSD at $p$=1, as infinite clusters can be broken easily



---

<div class="columns">

<iframe width="100%" height="100%" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/defence/DP_2D_stl.html" style="max-width:100% border: 1px solid #ccc" frameborder=0>
</iframe>

- In the 2D supercritical region, we let the system stabilize at $\rho^\text{stat}$

- Clusters cannot be broken apart with a single break

- Similar to site percolation, but with short range correlations


</div>


---

### New Critical Points

![width:1100](images/defence/colored_clusters_FSPL.png)


---

- Two new critical points!

- Increasing $p$, first *empty clusters* follow a power law

- Next, *filled clusters* follow a power law

- Exponent $\approx 1.85$ same for both!

![width:1100](images/defence/2D_newcriticalpoints.png)


---

### Dimensionality scaling

Critical points *swap positions* in 3D and beyond:

DP $\rightarrow$ Filled $\rightarrow$ Empty

![width:550](images/defence/DP_criticalpoints_dims_linear.png) $~$ ![width:550](images/defence/DP_criticalpoints_dims_normalized.png)


---

### Renormalized power laws

Renormalized to $x^{1.85}$. Site percolation $\approx$ 2.05

![width:1150px](images/defence/2D_renormalized_DP_criticalPoints.png)


---

## Conclusion

$\\$

- Model emphasizes **mutual feedback** between soil structure and ecosystem dynamics

- Spatial structure can solve the **parasite problem**

- Power-laws seen in the model can be explained through **directed percolation**.


---

# Additional Slides

---

#### Mean-Field: Higher $\rho$ always wins

<div class="columns">

<img src="images/feb29/parasite_meanfield.png" style="max-width: 100%; "></img>

<img src="images/feb29/parasite_lattice.png" style="max-width: 100%; "></img>

</div>


---

<iframe width="auto" height="1000px" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/twospec_samenutrient_lattice.html" style="border: 1px solid #ccc" frameborder=0>
</iframe>

---

#### Parasite Coexistence and  $\rho_2/\rho_1$

$~~~~~~~$ $\rho_2/\rho_1$=2 $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ $\rho_2/\rho_1$=4

![width:1100px](images/defence/parasite_rhofactor_coexistence.png)

---


### Multiple Nutrients
$\\$

- Have worms eat all other nutrients, besides their own
  - Worms simply oscillate in phase
  - Similar to single-species model

- Have worms eat nutrients in a cycle
  - 1 > 2 > 3 > 4 > 1, etc


---


<div class="columns">

<video src="images/feb29/4spec.mp4" style="max-width: 100%;" autoplay muted loop></video>

<video src="images/feb29/5spec.mp4" style="max-width: 100%;" autoplay muted loop></video>

</div>


---

Literature suggests **spiral waves**...

<img src="images/feb29/takeuchi_hogweg.png" style="max-width: 80%; "></img>

<span class="caption"> <a href="https://pubmed.ncbi.nlm.nih.gov/14604183/">Takeuchi and Hogweg, 2012</a> </span>


---

#### Why no Spirals?

<div class="columns">

- Assume we have a magical wave source.

- Waves ejected in the right order

- Soil needed for nutrient generation, and so waves **cannot propagate**!

<img src="images/defence/why_no_spirals_1.png" style="max-width: 45%; "></img>
<img src="images/defence/why_no_spirals_2.png" style="max-width: 45%; "></img>

</div>

---

##### After removing soil

<div class="columns">

<video src="images/feb29/nosoil_4spec.mp4" style="max-width: 100%;" autoplay muted loop></video>

<video src="images/feb29/nosoil_5spec.mp4" style="max-width: 100%;" autoplay muted loop></video>

</div>


---

### Spatial confinement

$\\$

- Assuming you're surrounded by soil, how do you propagate into it?

- *All species* must be present on the expansion boundary

- Expansion is hard!

---

<div class="columns">

<video src="images/feb29/4spec_boundary.mp4" style="max-width: 100%;" autoplay muted loop></video>

<video src="images/feb29/5spec_boundary.mp4" style="max-width: 100%;" autoplay muted loop></video>

</div>


---

### Powerlaw math
$\\$

<div class="columns">

<div>

$$
\begin{align*}
P_1(r)\; dr &= P_2(V) \; dV \\
P_1(r) &\sim P_2(V) \cdot r^2
\end{align*}
$$

Given that $P_2(V) \sim V^{-\tau}$ and $V\sim r^3$:

$$
\begin{align*}
P_1(r) &\sim V^{-\tau} \cdot r^2 \\
  &\sim r^{-3\tau} \cdot r^2 \\
  &\sim r^{2 - 3\tau}
\end{align*}
$$

</div>

<div>

$$
\begin{align*}
r^D &\sim \int P_1(r) dr \\
  &\sim \int r^{2-3\tau} dr \\
  &\sim r^{3-3\tau}
\end{align*}
$$

For $\tau\approx2.25$, $D\approx3.75$

</div>

</div>

---

#### Neighbour distances at DP point

![width:700px](images/defence/NbrDist_p=0.2873_L=1024.png)

---

### Site percolation

Different exponent of $\approx 2.05$
*Left:* same $\rho$ as DP, *Right:* Standard critical point

![width:1150px](images/defence/site_percolation_CSD.png)



