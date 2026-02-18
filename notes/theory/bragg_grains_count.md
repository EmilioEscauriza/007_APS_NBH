# Number of Diffracting Grains in Powder Film

## Context / Motivation

For single-crystal Bragg XPCS measurements on powder samples, we need to estimate how many crystallites simultaneously satisfy the Bragg condition in the illuminated volume. Too many grains → powder ring (no speckle contrast). The key constraint is the **maximum film thickness** before too many grains contribute.

With translation and rotation motors, finding grains is trivial. The binding constraint is the upper limit: the film must be thin enough that at any given beam position and sample angle, the number of simultaneously diffracting grains $N_\text{Bragg}$ stays below $N_\text{max}$.

## Assumptions

- Random grain orientations (powder sample)
- Grains are spherical with diameter $d$
- Packing fraction $\phi$ (volume fraction occupied by grains)
- Circular beam profile with diameter $D_\text{beam}$
- Surface reflection Bragg geometry (beam hits sample at angle $\theta_B$ from surface)
- Each grain has mosaic spread $\eta$ (angular distribution of crystallographic blocks)
- Optional: linear absorption coefficient $\mu$ for absorption-weighted effective count $N_\text{eff}$

## Derivation

### 1. Illuminated Volume

For a beam incident at Bragg angle $\theta_B$, the path length through a film of thickness $t$ is:

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

### 5. Absorption-Weighted Effective Number (Reflection Geometry)

The geometric count above assumes every grain in the volume contributes equally. For **reflection** (symmetric Bragg), two refinements are used:

**Depth distribution**  
Grains are uniformly distributed in depth $z \in [0, t]$. The average depth is $t/2$; the effective path length for a “typical” grain is $t/(2\sin\theta_B)$ rather than the full $t/\sin\theta_B$ if we were to weight by depth.

**Two-pass attenuation**  
A grain at depth $z$ is illuminated on the way in and the diffracted ray exits along a similar path. The intensity weight (incident × exit) is:

$$w(z) = \exp\!\left(-\mu \frac{z}{\sin\theta_B} - \mu \frac{z}{\sin\theta_B}\right) = \exp\!\left(-\frac{2\mu z}{\sin\theta_B}\right)$$

Define the decay constant (in depth units, e.g. µm⁻¹ if $z$ is in µm):

$$k = \frac{2\mu}{\sin\theta_B}$$

(with $\mu$ in cm⁻¹ and $z$ in the same length units as $t$; convert as needed so $k\,z$ is dimensionless). Then $w(z) = e^{-kz}$.

**Grains per unit depth** (along the surface normal):

$$n_z = \frac{\phi A_\text{beam}}{v_\text{grain}} \cdot \frac{1}{\sin\theta_B} \cdot P_\text{Bragg}$$

**Effective number of contributing grains** (absorption-weighted, participation-ratio style):

$$N_\text{eff}(t) = n_z \cdot \frac{2}{k} \cdot \tanh\!\left(\frac{k t}{2}\right)$$

- For $\mu \to 0$: $\tanh(kt/2) \approx kt/2$, so $N_\text{eff} \to n_z t = N_\text{Bragg}$ (geometric limit).
- For strong absorption: $N_\text{eff}$ saturates at $n_z \cdot (2/k)$, i.e. an attenuation-length-limited depth.

Speckle contrast uses this effective count:

$$\beta = \frac{1}{\max(N_\text{eff}, 1)}$$

**Maximum thicknesses** are found by solving $N_\text{eff}(t) = N_\text{max}$ or $N_\text{eff}(t) = 1$:

$$t = \frac{2}{k} \,\mathrm{arctanh}\!\left(\frac{N_\text{eff}\, k}{2 n_z}\right)$$

For low absorption, $t_\text{max,contrast} \approx 1/n_z + O(\mu^2)$, so the contrast thickness is almost independent of $\mu$ to first order.

## Result

**Geometric count** (all grains in volume count equally):

$$N_\text{Bragg} = \frac{\phi A_\text{beam} L}{v_\text{grain}} \times P_\text{Bragg}$$

with $L = t / \sin(\theta_B)$ and $P_\text{Bragg} = m_{hkl} \cos(\theta_B) \Delta\omega / 2$.

**Absorption-weighted effective count** (used for $\beta$ and $t_\text{max}$):

$$N_\text{eff}(t) = n_z \cdot \frac{2}{k} \cdot \tanh\!\left(\frac{k t}{2}\right), \qquad k = \frac{2\mu}{\sin\theta_B}, \qquad n_z = \frac{\phi A_\text{beam}}{v_\text{grain} \sin\theta_B} \, P_\text{Bragg}$$

**Maximum film thickness** (for $N_\text{eff} = N_\text{max}$):

$$t_\text{max,Nmax} = \frac{2}{k} \,\mathrm{arctanh}\!\left(\frac{N_\text{max}\, k}{2 n_z}\right)$$

(with geometric limit when $\mu \to 0$: $t_\text{max,Nmax} = N_\text{max} \cdot v_\text{grain} \sin\theta_B / (\phi A_\text{beam} P_\text{Bragg})$).

**Contrast threshold** (for $N_\text{eff} = 1$):

$$t_\text{max,contrast} = \frac{2}{k} \,\mathrm{arctanh}\!\left(\frac{k}{2 n_z}\right) \approx \frac{1}{n_z} \quad \text{(low } \mu \text{)}$$

## Key Dependencies

- **Grain size**: $N_\text{Bragg} \propto d^{-3}$ (cubic dependence!)
- **Beam size**: $N_\text{Bragg} \propto A_\text{beam} \propto D_\text{beam}^2$
- **Mosaic spread**: Dominates $\Delta\omega$ for real powders ($\eta \gg \delta_\text{div}, \delta_\text{Darwin}$)
- **Bragg angle**: Path length $L = t / \sin(\theta_B)$ increases at grazing angles

## Notes / Limitations

- Assumes uniform grain size (real powders have distributions)
- Mosaic spread $\eta$ is typically the dominant term in $\Delta\omega$
- For very small Bragg angles, the footprint becomes very elongated
- **Reflection geometry**: path length $L = t/\sin\theta_B$ and two-pass attenuation $w(z) = \exp(-2\mu z/\sin\theta_B)$ assume symmetric Bragg reflection (grazing incidence)
- For light-element materials (e.g. Na₂B₁₀H₁₀) at hard X-ray energies, $\mu$ is small so $N_\text{eff} \approx N_\text{Bragg}$ and $t_\text{max,contrast}$ is approximately independent of $\mu$ (to first order)
- With motors, there is no minimum thickness constraint (can scan to find grains)

## References

- Warren, B.E. "X-Ray Diffraction" (1969) — powder diffraction theory
- Guinier, A. "X-Ray Diffraction" (1994) — crystal optics and mosaic spread
- Authier, A. "Dynamical Theory of X-Ray Diffraction" (2001) — Darwin width
