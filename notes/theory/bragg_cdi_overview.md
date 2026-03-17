# Bragg Coherent Diffraction Imaging (BCDI) — Overview

*Monday, March 9, 2026*

---

## What is Bragg CDI?

Bragg CDI is a lens-free imaging technique that recovers the 3D shape and internal strain field of a nanocrystal from its coherent diffraction pattern measured around a Bragg peak. It yields real-space maps of electron density $\rho(\mathbf{r})$ and lattice displacement $\mathbf{u}(\mathbf{r})$ at nanometre resolution.

**Key review:** Robinson & Harder, *Nat. Mater.* **8**, 291 (2009).

---

## From rocking scan to 3D diffraction volume

### The Ewald sphere and angular slicing

A 2D area detector records a curved slice of reciprocal space defined by the Ewald sphere. In a rocking scan the sample is rotated through a small angular range $\Delta\theta$ about the Bragg condition. Each detector frame at angle $\theta_i$ captures a different 2D cut through the 3D intensity distribution $I(q_x, q_y, q_z)$ near the Bragg peak.

Stacking and interpolating these slices (with the correct angle-to-$\mathbf{q}$ mapping) assembles the full 3D coherent diffraction pattern. This is the preprocessing step before phase retrieval.

### Coordinate transform

For a diffractometer with sample angle $\omega$ and detector angle $2\theta$:

$$q_x = \frac{2\pi}{\lambda}\left[\cos(\delta)\sin(2\theta + \gamma) - \sin(\omega)\right]$$
$$q_y = \frac{2\pi}{\lambda}\left[\cos(\delta)\cos(2\theta + \gamma) - \cos(\omega)\right]$$
$$q_z = \frac{2\pi}{\lambda}\sin(\delta)$$

where $\gamma$ and $\delta$ are per-pixel angular offsets from the detector centre. In practice, libraries like `xrayutilities` or `cohere-core` handle this transform for specific beamline geometries.

**In this repo:** `rocking_curve_rsm()` in `workflow/workflow_lib.py` performs the 2D version of this mapping using `xrayutilities.HXRD`.

---

## The phase problem

The detector measures intensity, not amplitude:

$$I(\mathbf{Q}) = \left|\mathcal{F}\left[\psi(\mathbf{r})\right]\right|^2$$

where $\psi(\mathbf{r}) = \rho(\mathbf{r})\,e^{i\,\mathbf{G}_{hkl}\cdot\mathbf{u}(\mathbf{r})}$ is the complex object function. The Fourier magnitudes $|\mathcal{F}[\psi]|$ are known from $\sqrt{I}$, but the phases are lost. Phase retrieval algorithms recover these phases iteratively.

---

## Phase retrieval algorithms

All iterative algorithms alternate between two constraint sets applied in conjugate domains:

### Fourier-space constraint

Replace the magnitudes of the current estimate with the measured values, keeping the estimated phases:

$$\Psi'(\mathbf{Q}) = \sqrt{I(\mathbf{Q})} \; \frac{\Psi(\mathbf{Q})}{|\Psi(\mathbf{Q})|}$$

### Real-space constraint

Enforce that the object is confined to a finite support region $S$ (the set of voxels expected to contain the particle).

The algorithms differ in how they handle voxels outside the support:

| Algorithm | Outside support | Notes |
|-----------|----------------|-------|
| **ER** (Error Reduction) | $\psi_n = 0$ | Guaranteed convergence; stagnates at local minima |
| **HIO** (Hybrid Input-Output) | $\psi_n = \psi_{n-1} - \beta\,\psi'_n$ | Escapes local minima via feedback ($\beta \approx 0.9$) |
| **RAAR** (Relaxed Averaged Alternating Reflections) | Weighted average of ER and HIO updates | Better convergence theory than HIO |

**Fienup, *Appl. Opt.* 21, 2758 (1982)** — original HIO paper.

### Shrink-wrap support update

The support $S$ is not always known a priori. Shrink-wrap (Marchesini et al., 2003) periodically updates it by:

1. Convolving $|\psi_n(\mathbf{r})|$ with a Gaussian kernel.
2. Thresholding at a fraction of the maximum.

This allows the algorithm to discover the particle shape automatically.

### Typical reconstruction recipe

```
200 iterations HIO  (β = 0.9)
 50 iterations ER   (polish)
Shrink-wrap every 20 iterations
Multiple random starts → select by lowest error metric
```

**Clark et al., *Science* 341, 56 (2013)** — full pipeline example from rocking scan to 3D strain map.

---

## Strain from phase

The recovered complex object is:

$$\psi(\mathbf{r}) = \rho(\mathbf{r})\,e^{i\,\phi(\mathbf{r})}$$

- $|\psi(\mathbf{r})| = \rho(\mathbf{r})$: electron density — gives the **particle shape**.
- $\phi(\mathbf{r}) = \mathbf{G}_{hkl} \cdot \mathbf{u}(\mathbf{r})$: phase — encodes the **displacement field** projected onto the scattering vector.

The projected strain is:

$$\varepsilon_{hkl}(\mathbf{r}) = \frac{\nabla\phi(\mathbf{r})}{|\mathbf{G}_{hkl}|}$$

A single reflection gives one component of displacement. Multiple reflections (measured at different $\mathbf{G}_{hkl}$) can be combined to recover the full 3D strain tensor, though this requires separate rocking scans for each reflection.

**In this repo:** the forward model in `modelling/bragg_forward_model.py` computes the forward direction of this relationship:

$$I(\mathbf{Q}) = \left|\text{FFT}\left[\rho(\mathbf{r})\,e^{i\,\mathbf{G}_{hkl}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{r}}\right]\right|^2$$

Phase retrieval inverts this to recover $\rho$ and $\boldsymbol{\varepsilon}\cdot\mathbf{r}$ from $I$.

---

## Oversampling requirement

Phase retrieval requires that the diffraction pattern is oversampled relative to the Nyquist frequency — i.e., the coherent speckle size must be resolved by the detector pixels.

The oversampling ratio along each dimension is:

$$\sigma = \frac{\lambda \, z}{D \, \Delta p}$$

where $z$ is the sample-to-detector distance, $D$ is the object size, and $\Delta p$ is the pixel size. For $\sigma \geq 2$ in each dimension, the phase problem is (generically) uniquely solvable in 2D and 3D.

**For this experiment** (from `notes/experimental_parameters.md`):
- $\lambda = 1.00$ Å (12.4 keV), $z = 2.2$ m, $\Delta p = 55$ µm
- For a $D = 300$ nm grain: $\sigma = (1.00 \times 10^{-10} \times 2.2) / (300 \times 10^{-9} \times 55 \times 10^{-6}) \approx 13$

This is well oversampled — each coherent speckle spans about 13 detector pixels.

---

## Connection to XPCS

| Technique | Measures | Domain | Time |
|-----------|----------|--------|------|
| **Bragg CDI** | $\rho(\mathbf{r})$ and $\mathbf{u}(\mathbf{r})$ | Real space (3D) | Single snapshot |
| **Bragg XPCS** | $g_2(\tau)$ and TTC$(t_1, t_2)$ | Reciprocal space | Time series |

They are complementary views of the same physics:

- **CDI** tells you *what the grain looks like*: shape, facets, dislocations, strain distribution at a given moment.
- **XPCS** tells you *how it evolves*: relaxation rates, oscillation periods, ageing.

The speckle fluctuations measured by XPCS arise from changes in $\psi(\mathbf{r}, t)$. If a displacement field $\mathbf{u}(\mathbf{r})$ evolves in time, the corresponding phase $\phi(\mathbf{r}, t) = \mathbf{G}\cdot\mathbf{u}(\mathbf{r},t)$ changes, causing the speckle pattern to decorrelate. The TTC oscillation periods analysed in `notes/theory/single_frequency_ttc_model.md` reflect periodic strain dynamics within the illuminated grain.

Time-resolved CDI (chrono-CDI) reconstructs $\psi(\mathbf{r})$ at multiple time windows within a single measurement, bridging the two techniques. This is one of the features of `cohere-core`.

---

## Reconstruction tool: cohere-core

The [cohere-core](https://pypi.org/project/cohere-core/) package (Argonne, B. Frosik & R. Harder) provides a complete Bragg CDI pipeline:

| Module | Function |
|--------|----------|
| Preprocessing | Data formatting, centering, cropping, alien removal (AutoAlien1) |
| Reconstruction | ER, HIO, RAAR, shrink-wrap, Genetic Algorithm (GA) for multi-start |
| AI initial guess | AutoPhaseNN (Yao et al.) for learned starting points |
| Chrono-CDI | Reduced oversampling for time-resolved reconstruction (Ulvestad et al.) |
| GPU support | numpy, cupy, torch backends |

Documentation: https://cohere.readthedocs.io/

Install: `pip install cohere-core`

---

## Key references

1. Robinson, I. & Harder, R. Coherent X-ray diffraction imaging of strain at the nanoscale. *Nat. Mater.* **8**, 291–298 (2009).
2. Fienup, J. R. Phase retrieval algorithms: a comparison. *Appl. Opt.* **21**, 2758–2769 (1982).
3. Clark, J. N. et al. Three-dimensional imaging of dislocation propagation during crystal growth and dissolution. *Science* **341**, 56–59 (2013).
4. Ulvestad, A. et al. Topological defect dynamics in operando battery nanoparticles. *Nano Lett.* **15**, 4066–4070 (2015).
5. Marchesini, S. et al. X-ray image reconstruction from a diffraction pattern alone. *Phys. Rev. B* **68**, 140101 (2003). [Shrink-wrap]
