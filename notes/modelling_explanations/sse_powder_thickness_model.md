# sse_powder_thickness_model.py

Estimate the **maximum** powder film thickness for single-crystal Bragg XPCS.

## Purpose

With translation and rotation motors, finding a diffracting grain is trivial (scan until one appears). The binding constraint is the **upper limit**: the film must be thin enough that at any given beam position and sample angle, the number of simultaneously diffracting grains $N_{\text{Bragg}}$ stays below $N_{\max}$ — otherwise spots merge into a powder ring and speckle contrast is lost.

## Physics

$$L = t / \sin(\theta_B)$$

$$N_{\text{Bragg}} = \frac{\phi \cdot A_{\text{beam}} \cdot L}{v_{\text{grain}}} \cdot \frac{m_{hkl} \cos(\theta_B) \Delta\omega}{2}$$

$$t_{\max} = \text{thickness at which } N_{\text{Bragg}} = N_{\max}$$

where:
- $\Delta\omega$ is the effective rocking-curve width (mosaic ⊕ Darwin ⊕ beam divergence)
- $v_{\text{grain}} = (4/3)\pi(d/2)^3$
- Beam footprint on sample surface is an ellipse with short axis $D$ and long axis $D/\sin(\theta_B)$

## Transmission

$$T = \exp(-2 \phi \mu L), \quad L = t/\sin(\theta_B) \text{ [cm]}$$

where $\phi$ is the packing fraction. Double-pass gives $T_{\text{single}}^2$ per path.

## Usage

Run interactively or import functions for calculations.
