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



## PhD Research Plan

<br>

**Riz Fernando Noronha**
**Niels Bohr Institute**


---

### Methodology

<br>

- Simulations + Analytical work
  <br>
  - Analytical solutions where possible
  - Simulations to explore complex scenarios

---

### Guiding Principles

<br>

1. **Simple** models
   - Strip phenemena down!
   - Few parameters
<br>
1. **Generalizable** models
   - Behaviour across different length scales

---

### Project 1: Evolution

<br>

- Robustness and Plasticity
  - very _generic_ concepts, true across length scales

<br>

**Goal:** To make a _simple_ model with this behaviour

---

![bg right:40% fit](images/PhD/genetic_algorithm.png)

**Genetic Algorithms:**

<br>

- Built similar to GRNs

- Take top performers and mutate

- Robustness emerges (depending on noise)

<span class='caption'><a href='https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000434'>K. Kaneko (2007)</a></span>

---

![bg right fit](images/PhD/highly_optimized_tolerance.png)


**Highly Optimized Tolerance (HOT)**

<span class='caption'><a href='https://journals.aps.org/pre/abstract/10.1103/PhysRevE.60.1412'>Carlson, J. M., & Doyle, J. (1999)</a></span>

<br>  

- Cellular automata

- Suggests optimal behavior is critical

- Robustness? 

---

### Project 2: SOC in Neural Avalanches

<br>

**Goal:** Explain the SOC behaviour

<br>

<div class='columns'>
Periods of inactivity followed by multiple neuron activations

<img src="images/PhD/neuronal_avalanche.png" style="max-width: 100%; "></img>
<span class='caption'><a href='https://www.jneurosci.org/content/23/35/11167.short'>Beggs, J. M., & Plenz, D. (2003)</a></span>
</div>

---

![bg right:30% fit](images/PhD/neuronal_avalanche_periodic.png)

- **Avalanche models** suggested
  - data has _repeating patterns_
    <span class='caption'><a href='https://pubmed.ncbi.nlm.nih.gov/15175392/'>Beggs, J. M., & Plenz, D. (2004)</a></span>
<br>
- "Percolation" like **network model** instead
  - needs to be tuned to criticality
  
---

<div class='columns'>

"Doping" the cortex
<img src="images/PhD/neuronal_avalanche_doped.png" style="max-width: 100%; "></img> 
<span class='caption'> <a href='https://link.springer.com/article/10.1140/epjst/e2012-01575-5'>Plenz, D. (2012)</a> </span>


Development over time
<img src="images/PhD/avalanche_development.png" style="max-width: 70%; "></img> 
<span class='caption'> <a href='https://www.frontiersin.org/articles/10.3389/fphy.2021.639389/full'>Plenz, D., Ribeiro, T. L., Miller, S. R., Kells, P. A., Vakili, A., & Capek, E. L. (2021)</a> </span>

</div>
