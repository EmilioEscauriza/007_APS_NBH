# Number of Diffracting Grains in Powder Film

## Context / Motivation

For single-crystal Bragg XPCS measurements on powder samples, we need to estimate how many crystallites simultaneously satisfy the Bragg condition in the illuminated volume. Too many grains → powder ring (no speckle contrast). The key constraint is the **maximum film thickness** before too many grains contribute.

With translation and rotation motors, finding grains is trivial. The binding constraint is the upper limit: the film must be thin enough that at any given beam position and sample angle, the number of simultaneously diffracting grains $N_\text{Bragg}$ stays below $N_\text{max}$.

## Assumptions

- Random grain orientations (powder sample)
- Grains are spherical with diameter $d$
- Packing fraction $\phi$ (volume fraction occupied by grains)
- Circular beam profile with diameter $D_\text{beam}$
- Grazing incidence Bragg geometry (beam hits sample at angle $\theta_B$ from surface)
- Each grain has mosaic spread $\eta$ (angular distribution of crystallographic blocks)

## Derivation

### 1. Illuminated Volume

For a beam incident at grazing angle $\theta_B$ (Bragg geometry), the path length through a film of thickness $t$ is:

$$L = \frac{t}{\sin(\theta_B)}$$

The beam cross-sectional area (perpendicular to beam direction) is:

$$A_\text{beam} = \pi \left(\frac{D_\text{beam}}{2}\right)^2$$

So the **illuminated volume** is:

$$V_\text{illum} = A_\text{beam} \times L = A_\text{beam} \times \frac{t}{\sin(\theta_B)}$$

The beam footprint on the sample surface is an ellipse with:
- Short axis: $D_\text{beam}$
- Long axis: $D_\text{beam} / \sin(\theta_B)$

### 2. Total Number of Grains in Illuminated Volume

Volume per grain (assuming spherical grains):

$$v_\text{grain} = \frac{4}{3}\pi \left(\frac{d}{2}\right)^3 = \frac{\pi d^3}{6}$$

Total number of grains in illuminated volume:

$$N_\text{total} = \frac{\phi V_\text{illum}}{v_\text{grain}} = \frac{\phi A_\text{beam} L}{v_\text{grain}} = \frac{\phi A_\text{beam} t}{v_\text{grain} \sin(\theta_B)}$$

### 3. Probability a Grain Satisfies the Bragg Condition

For a randomly oriented crystallite, the normal to the $(hkl)$ planes is uniformly distributed on the unit sphere. The Bragg condition is satisfied when this normal lies within a narrow angular band around the incident beam direction.

The **effective rocking-curve width** $\Delta\omega$ includes contributions from:
- Mosaic spread $\eta$ (grain internal misorientation)
- Beam divergence $\delta_\text{div}$
- Intrinsic Darwin width $\delta_\text{Darwin}$ (perfect crystal)

Combined in quadrature:

$$\Delta\omega = \sqrt{\eta^2 + \delta_\text{div}^2 + \delta_\text{Darwin}^2}$$

For a single set of $(hkl)$ planes, the solid angle on the unit sphere that satisfies Bragg is:

$$\Delta\Omega = 2\pi \cos(\theta_B) \Delta\omega$$

So the probability for one set of planes:

$$P_\text{single} = \frac{\Delta\Omega}{4\pi} = \frac{\cos(\theta_B) \Delta\omega}{2}$$

Accounting for **multiplicity** $m_{hkl}$ (number of symmetry-equivalent reflections):

$$P_\text{Bragg} = \frac{m_{hkl} \cos(\theta_B) \Delta\omega}{2}$$

### 4. Expected Number of Diffracting Grains

Combining the grain count with the Bragg probability:

$$N_\text{Bragg} = N_\text{total} \times P_\text{Bragg}$$

Substituting:

$$\begin{align}
N_\text{Bragg} &= \frac{\phi A_\text{beam} t}{v_\text{grain} \sin(\theta_B)} \times \frac{m_{hkl} \cos(\theta_B) \Delta\omega}{2} \\
&= \frac{\phi A_\text{beam} t \, m_{hkl} \cos(\theta_B) \Delta\omega}{2 v_\text{grain} \sin(\theta_B)}
\end{align}$$

Using $\cos(\theta_B) / \sin(\theta_B) = \cot(\theta_B)$:

$$N_\text{Bragg} = \frac{\phi A_\text{beam} t \, m_{hkl} \cot(\theta_B) \Delta\omega}{2 v_\text{grain}}$$

## Result

**Expected number of simultaneously diffracting grains:**

$$N_\text{Bragg} = \frac{\phi A_\text{beam} L}{v_\text{grain}} \times \frac{m_{hkl} \cos(\theta_B) \Delta\omega}{2}$$

where $L = t / \sin(\theta_B)$ is the path length through the film.

**Maximum film thickness** (for $N_\text{Bragg} = N_\text{max}$):

$$t_\text{max,Nmax} = \frac{N_\text{max} v_\text{grain} \sin(\theta_B)}{\phi A_\text{beam} P_\text{Bragg}}$$

where $P_\text{Bragg} = m_{hkl} \cos(\theta_B) \Delta\omega / 2$.

**Contrast threshold** (for $N_\text{Bragg} = 1$, perfect single-grain contrast):

$$t_\text{max,contrast} = \frac{v_\text{grain} \sin(\theta_B)}{\phi A_\text{beam} P_\text{Bragg}}$$

## Key Dependencies

- **Grain size**: $N_\text{Bragg} \propto d^{-3}$ (cubic dependence!)
- **Beam size**: $N_\text{Bragg} \propto A_\text{beam} \propto D_\text{beam}^2$
- **Mosaic spread**: Dominates $\Delta\omega$ for real powders ($\eta \gg \delta_\text{div}, \delta_\text{Darwin}$)
- **Bragg angle**: Path length $L = t / \sin(\theta_B)$ increases at grazing angles

## Notes / Limitations

- Assumes uniform grain size (real powders have distributions)
- Mosaic spread $\eta$ is typically the dominant term in $\Delta\omega$
- For very small Bragg angles, the footprint becomes very elongated
- Absorption is negligible for light-element materials like Na₂B₁₀H₁₀ at hard X-ray energies
- With motors, there is no minimum thickness constraint (can scan to find grains)

## References

- Warren, B.E. "X-Ray Diffraction" (1969) — powder diffraction theory
- Guinier, A. "X-Ray Diffraction" (1994) — crystal optics and mosaic spread
- Authier, A. "Dynamical Theory of X-Ray Diffraction" (2001) — Darwin width
