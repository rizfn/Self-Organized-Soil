---
marp: true
theme: uncover
math: mathjax
paginate: true
_paginate: skip
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
        }

---


![bg right:42% brightness:1.5 saturate:1.5](images/soil.png)

# Self-Organised .....Soil (??)

<br>
Riz Fernando Noronha

Supervised by Kim Sneppen and Kunihiko Kaneko


---

<div class='columns'>


- Soil particle-sizes follow a power-law

- There is a correlation between the fractal dimension and the biodiversity of the soil

- *Hypothesis:* Biology contributes to the fractal structure

<img src="images/feb29/tyler_wheatcraft.png" style="max-width: 90%;"></img>
<span class="caption"><a href="https://acsess.onlinelibrary.wiley.com/doi/abs/10.2136/sssaj1989.03615995005300040001x">Tyler and Wheatcraft, 1989</a></span>

</div>

---

### Parameters

- $\sigma$: Soil filling rate
- $\theta$: Death rate of worms
- $\mu$: Nutrient generation rate
- $\rho$: Reproductive rate of worms

<br>

> *One is a great number!*

---

### Algorithm

<div class="columns">

<img src="images/feb29/algo_network.png" style="max-width: 100%; "></img>

<img src="images/feb29/algo_grid.png" style="max-width: 100%; "></img>

</div>

---

<div class="columns">

![h:500px](images/feb29/lattice3D.png)

- Worm create nutrients *in their vicinity*

- Mutual co-existence

- Can be thought of as a predator-prey, as the worms consume soil in a 2-step process

</div>

---

<iframe width="auto" height="1000px" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/nutrient_lattice.html"style="border: 1px solid #ccc" frameborder=0>
</iframe>

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

<iframe width="auto" height="1000px" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/nutrient_meanfield.html"style="border: 1px solid #ccc" frameborder=0>
</iframe>

---

### Survival in lattice vs Meanfield

![w:1150px](images/feb29/raster_row.png)

---

### Two Species

- Add a second species, and make it asymmetric in terms of:
  - $\mu$, nutrient generation rate
  - $\rho$, reproduction rate
<br>
- $\mu=0$ is a *parasite*.

---

#### Mean-Field: Parasite Problem

<br>

![w:800px](images/feb29/competitive_exclusion.png)

- Worms don't care who creates the nutrient
- A slight advantage mean you dominate
- **Competitive Exclusion:** Higher $\rho$ always wins

---

#### Mean-Field: Higher $\rho$ always wins

<div class="columns">

<img src="images/feb29/parasite_meanfield.png" style="max-width: 100%; "></img>

<img src="images/feb29/parasite_lattice.png" style="max-width: 100%; "></img>

</div>


---

<iframe width="auto" height="1000px" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/twospec_samenutrient_lattice.html"style="border: 1px solid #ccc" frameborder=0>
</iframe>

---

### Multiple Nutrients
<br>

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

<img src="images/feb29/soil_expansion_1.png" style="max-width: 50%; "></img>
<img src="images/feb29/soil_expansion_2.png" style="max-width: 50%; "></img>

</div>

---

##### After removing soil

<div class="columns">

<video src="images/feb29/nosoil_4spec.mp4" style="max-width: 100%;" autoplay muted loop></video>

<video src="images/feb29/nosoil_5spec.mp4" style="max-width: 100%;" autoplay muted loop></video>

</div>


---

### Spatial confinement

<br>

- Assuming you're surrounded by soil, how do you propagate into it?

- *All species* must be present on the expansion boundary

- Expansion is hard!

---

<div class="columns">

<video src="images/feb29/4spec_boundary.mp4" style="max-width: 100%;" autoplay muted loop></video>

<video src="images/feb29/5spec_boundary.mp4" style="max-width: 100%;" autoplay muted loop></video>

</div>


---

### Power Laws

- For certain parameter values, we do see power-law distributed "particle sizes".

<div class="columns">

<img src="images/feb29/CSD_sigma_1_theta_0.038.png" style="max-width: 100%; "></img>

- Size ~ Cluster size

- Potentially, _oscillations_ span across the critical point

- More work is needed! 

</div>



---

<iframe width="auto" height="1000px" src="https://rizfn.github.io/Self-Organized-Soil/visualizations/multispec_nosoil_lattice3D_anim.html" frameborder=0>
</iframe>

