---
marp: true
theme: uncover
math: mathjax
paginate: true
_paginate: skip
# backgroundImage: url('images/defence/frontpage/ku-outline-seal.png')
# backgroundImage: url('images/PhD/KU5.png')
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

- **Soil porosity** appears to increase

- *Larger particles* (>2μm) of soil are observed

- Perhaps biologically induced **soil-aggregation**?


<div class="columns">

<img src="images/defence/feeney_porosity.png" style="max-width: 100%;"></img>

<img src="images/csl_paris/feeney_soil_cluster_size.png" style="max-width: 100%;"></img>

</div>

<span class="caption"><a href="https://www.pnas.org/doi/full/10.1073/pnas.2304663120">Feeney et al 2006</a></span>

---

### Feedback Loop?

![width:1100px](images/EPFL_latsis/feedback_loop.png)


---

### Algorithm

<div class="columns">

<img src="images/csl_paris/SENM_schematic.png" style="max-width: 90%; "></img>

- Microbes convert soil to nutrients
- Microbes use nutrients to reproduce
- Soil 'grows' from empty and nutrients
- Predator-prey with time-delay

</div>


---

### Cellular Automaton Model

$\\$

<div class="columns">

- Stochastic Sequential Algorithm

- Microbe create nutrients *in their vicinity*

- Microbes thus help each other survive

<img src="images/defence/algo_grid.png" style="max-width: 75%; "></img>

</div>

---

<iframe width="100%" height="100%" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/defence/nutrient_lattice_density.html" style="border: 1px solid #ccc" frameborder=0>
</iframe>


---

### Survival in lattice vs Meanfield

![w:1000](images/EPFL_latsis/rasterscan_attractors.png)


---

### Soil particle sizes

![width:800](images/defence/soil_cluster_size36.png)

Nearest neighbour connected components for clusters:
$\implies$ both are a cluster of size 36!


---

<iframe width="100%" height="100%" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/defence/nutrient_lattice_density.html" style="border: 1px solid #ccc" frameborder=0>
</iframe>

---

### Cluster size distributions

![width:1000](images/hiroshima/empty_clusters_paper.png)
![width:1000](images/hiroshima/soil_clusters_paper.png)

---

## Nutrient Maximization

![width:1100](images/csl_paris/nutrientmax_soilboundaries.png)


---

## Conclusion


![width:900px](images/EPFL_latsis/conclusion_2.png)

---

### Acknowledgements

$\\$

<div class="columns">

<img src="images/EPFL_latsis/kunihiko_kaneko.jpg" style="max-height: 20%; "></img>
Kunihiko Kaneko

<img src="images/EPFL_latsis/kim_sneppen.jpg" style="max-height: 20%; "></img>
Kim Sneppen

</div>



---

# Additional Slides

---

### Mean-Field Equations

$$
\definecolor{darkergrey}{rgb}{0.35, 0.35, 0.35}
\begin{align}
\frac{\mathrm{d}\textcolor{brown}{S}}{\mathrm{d}t} &= \sigma \textcolor{brown}{S} (\textcolor{darkergrey}{E} + \textcolor{teal}{N}) - \textcolor{green}{M} \textcolor{brown}{S} \\
\frac{\mathrm{d}\textcolor{darkergrey}{E}}{\mathrm{d}t} &= \theta \textcolor{green}{M} - \sigma \textcolor{brown}{S} \textcolor{darkergrey}{E} \\
\frac{\mathrm{d}\textcolor{teal}{N}}{\mathrm{d}t} &= \textcolor{green}{M} \textcolor{brown}{S} - \textcolor{green}{M} \textcolor{teal}{N} - \sigma \textcolor{brown}{S} \textcolor{teal}{N} \\
\frac{\mathrm{d}\textcolor{green}{M}}{\mathrm{d}t} &= \textcolor{green}{M} \textcolor{teal}{N} - \theta \textcolor{green}{M}
\end{align}
$$

---

<iframe width="100%" height="100%" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/nutrient_meanfield_attractors.html" style="border: 1px solid #cccccc" frameborder=0>
</iframe>


---


<iframe width="100%" height="100%" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/defence/nutrient_lattice_3D_density.html" style="border: 1px solid #ccc" frameborder=0>
</iframe>

---

### Survival in lattice vs Meanfield

![w:1150](images/defence/rasterscan_attractors.png)

---

![bg left:65% fit](images/csl_paris/3d_vs_mf_timeseries.png)

Spatial structure **damps** oscillations


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

### Investigating Cluster Size Distributions

<div class="columns">

<iframe width="100%" height="100%" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/defence/DP_2D_stl.html" style="max-width:100% border: 1px solid #ccc" frameborder=0>
</iframe>

- In the 2D supercritical region, we let the system stabilize at $\rho^\text{stat}$

- Take slice at *constant time*, and look at the 2D structure

- Similar to site percolation, but with short range correlations


</div>

---

### Critical Points

![width:1100](images/defence/colored_clusters_FSPL.png)


---

- Site percolation critical points

- Increasing $p$, first *empty clusters* follow a power law

- Next, *filled clusters* follow a power law

- Exponent appears to be $\approx 1.85$

![width:1100](images/defence/2D_newcriticalpoints.png)


---

### Dimensionality scaling

Critical points *swap positions* in 3D and beyond:

DP $\rightarrow$ Filled $\rightarrow$ Empty

![width:550](images/defence/DP_criticalpoints_dims_linear.png) $~$ ![width:550](images/defence/DP_criticalpoints_dims_normalized.png)


---

### Two Species: Parasite

$$\\$$

<div class="columns">

- Add a second species, a parasite
- Parasites **cannot create** nutrients
- Parasites live longer (die at rate $\theta_2 ,< \theta_1$) to compensate 

<img src="images/csl_paris/parasite_schematic.png" style="max-width: 85%; "></img>

</div>


---

#### Mean-Field: Parasite Problem

$\\$

- Microbes don't care who creates the nutrient
- A slight advantage mean you dominate
- **Coexistence is impossible!**
  *Competitive exclusion*: Lower $\theta$ always wins


---

<iframe width="100%" height="100%" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/defence/twospec_samenutrient_lattice_st_density.html" style="border: 1px solid #ccc" frameborder=0>
</iframe>

---

### Varying parasite strength

![width:1100px](images/csl_paris/parasite_coexistence_thetafactor_248.png)


---


### Two Symbiotes


![bg fit left:64.5%](images/csl_paris/twospec_symbiotes.png)

- Two species, two nutrients

- Eat the other's nutrient to survive

---

### Nutrient Correlation in other cases

$\\$

**Parasites:**

- Power law $=$ long range correlations $\sim$ Meanfield

- Meanfield is **bad**, so power law is avoided?

**3D:**

- Isotropic model, real soil is siginificantly anisotropic

---

<iframe width="100%" height="100%" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/twospec_samenutrient_lattice.html" style="border: 1px solid #ccc" frameborder=0>
</iframe>

---

### Multiple Nutrients
$\\$

- Have microbes eat all other nutrients, besides their own
  - Microbes simply oscillate in phase
  - Similar to single-species model

- Have microbes eat nutrients in a cycle
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
