# How the heterodyne TTC plot is calculated

This note explains how the two-time correlation $c_2(t_1, t_2)$ is computed for the **heterodyne** (two-component) transport-coefficient model used in the plots. It is written so the reader does not need access to any code. We only describe the case where the **transport coefficient and fractions are time-dependent** (so there is a ramp in the plot) and where the **velocity is periodic in time**.

References: He et al. 2024 PNAS, “Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter” (Supporting Information Eq. S-95).

---

## 1. Time grid

Time is discretized as $t_0, t_1, \ldots, t_{N-1}$ with constant spacing $\Delta t$ (e.g. 0.5 s). The plot is a 2D map over $(t_1, t_2)$: for each pair of times we compute one value $c_2(t_1, t_2)$. So we work on an $N\times N$ grid.

---

## 2. Integrals between two times

Many steps below need the integral of a time-dependent quantity $y(s)$ between two times $t_1$ and $t_2$. That integral is computed as follows.

- First, build the **cumulative** integral from the start of the run up to each time $t_k$:
  $$
  F(t_k) = \int_{t_0}^{t_k} y(s)\,ds
  $$
  using the trapezoidal rule on the discrete $t$ array.

- Then, for any $(t_1, t_2)$, the integral from $t_1$ to $t_2$ is
  $$
  \int_{t_1}^{t_2} y(s)\,ds = F(t_2) - F(t_1),
  $$
  with $F$ evaluated by linear interpolation when $t_1$ or $t_2$ fall between grid points.

This is done for each grid point $(t_1, t_2)$ so we get 2D arrays of integrals.

---

## 3. Heterodyne formula for $c_2$

The model has two components: a **static** (reference) fraction $x_r(t)$ and a **flowing** fraction $x_s(t)$, with $x_r(t) + x_s(t) = 1$. The flowing component has mean velocity $v(t)$ and both components have a common transport coefficient $J(t)$. The second-order two-time correlation is

$$
c_2(t_1, t_2) = 1 + \frac{\beta}{f^2}\,
\exp\bigl(-q^2 \mathcal{J}(t_1,t_2)\bigr)\,
\Bigl[
  (x_{r1}x_{r2})^2 + (x_{s1}x_{s2})^2
  + 2\,x_{r1}x_{r2}x_{s1}x_{s2}\,
  \cos\bigl(q\cos\phi\;\mathcal{V}(t_1,t_2)\bigr)
\Bigr].
$$

Notation:

- $\beta$: speckle contrast (0–1).
- $q$: magnitude of the scattering vector; $\phi$: angle between $\vec{q}$ and the flow direction.
- $x_{r1} = x_r(t_1)$, $x_{r2} = x_r(t_2)$, $x_{s1} = x_s(t_1)$, $x_{s2} = x_s(t_2)$.
- **Normalization:**
  $$
  f^2 = (x_{s1}^2 + x_{r1}^2)(x_{s2}^2 + x_{r2}^2).
  $$
- **Delay-based decay (so the diagonal is brightest):**
  $$
  \mathcal{J}(t_1,t_2) = \int_{\min(t_1,t_2)}^{\max(t_1,t_2)} J(s)\,ds.
  $$
  So we use the integral of $J$ over the **delay** $\tau = |t_2 - t_1|$, not the signed interval. That makes the decay symmetric: on the diagonal $t_1 = t_2$ the integral is zero and the exponential is 1; moving away from the diagonal it increases and the exponential decreases.

- **Phase of the stripes:**
  $$
  \mathcal{V}(t_1,t_2) = \int_{t_1}^{t_2} v(s)\,ds.
  $$
  This is the **signed** integral of the velocity from $t_1$ to $t_2$. It sets the phase of the $\cos$ term and thus the stripe pattern.

So for every $(t_1, t_2)$ we:

1. Compute $\mathcal{J}(t_1,t_2)$ and $\mathcal{V}(t_1,t_2)$ using the integral procedure above (with $y = J$ and $y = v$ respectively; for $\mathcal{J}$ we use the absolute value of the integral over the interval from $\min(t_1,t_2)$ to $\max(t_1,t_2)$).
2. Evaluate $x_r$ and $x_s$ at $t_1$ and $t_2$.
3. Form $f^2$ and the bracket, then apply the formula.

---

## 4. Time-dependent functions used in the plot

Here we only describe the **time-dependent** choices (ramp) for $J$ and $x_s$, and the **periodic** choice for $v$.

### 4.1 Velocity $v(t)$ — periodic

$$
v(t) = v_{\text{mean}}\,\bigl(1 + a\,\cos(2\pi t / T)\bigr).
$$

- $v_{\text{mean}}$ is a reference velocity (e.g. set from a chosen stripe period and $q$: $v_{\text{mean}} = 2\pi/(q\cdot T_{\text{stripe}})$).
- $a$ is an amplitude fraction (e.g. 0.2), with $0 < a < 1$ so $v(t)$ stays positive.
- $T$ is the period of the modulation in time (e.g. 150 s).

So the velocity oscillates in time; that makes the stripe frequency along antidiagonals vary with the start point on the diagonal.

### 4.2 Transport coefficient $J(t)$ and flowing fraction $x_s(t)$ — time-dependent (ramp)

We define an auxiliary time variable running backward:

$$
s = t_{\max} - t,
$$

where $t_{\max}$ is the last time in the window. Then:

**Flowing fraction:**

$$
x_s(t) = 0.25 + 0.45\,\bigl(1 - e^{-s/(0.25\,t_{\max})}\bigr).
$$

So $x_s$ varies smoothly with “reversed” time $s$; the static fraction is $x_r(t) = 1 - x_s(t)$.

**Transport coefficient:**

$$
J(t) = J_0\,\exp\bigl(-s/(0.6\,t_{\max})\bigr), \quad J_0 = 1.5.
$$

So $J$ also depends on $s$ and thus on $t$. Because both $J(t)$ and $x_s(t)$ vary with $t$, the prefactor and the bracket in the $c_2$ formula depend on where we are in the $(t_1, t_2)$ plane, which produces the **ramp** (systematic brightness change perpendicular to the diagonal). Using $s = t_{\max} - t$ reverses the direction of that ramp compared to using $t$ directly.

---

## 5. Symmetry of the displayed plot

The formula above is evaluated on the full $(t_1, t_2)$ grid. For display, the half with $t_1 > t_2$ is kept as computed, and the half with $t_1 < t_2$ is filled by symmetry: the value at $(t_1, t_2)$ is set equal to the value at $(t_2, t_1)$. So the final plot is symmetric about the line $t_1 = t_2$.

---

## Summary

- **Grid:** $N$ time points; $c_2$ is computed on an $N\times N$ $(t_1, t_2)$ grid.
- **Integrals:** Cumulative trapezoidal $F(t)$, then $\int_{t_1}^{t_2} y\,ds = F(t_2)-F(t_1)$ (with $\mathcal{J}$ taken as the integral over the delay interval so the diagonal is the maximum).
- **Heterodyne formula:** $c_2 = 1 + (\beta/f^2)\,\exp(-q^2\mathcal{J})\,\times$ bracket, with bracket containing $x_r,x_s$ at $t_1,t_2$ and $\cos(q\cos\phi\;\mathcal{V})$.
- **This plot:** $v(t)$ is periodic; $J(t)$ and $x_s(t)$ are time-dependent via $s = t_{\max}-t$, giving stripes plus a ramp. The plot is then symmetrized about $t_1 = t_2$.
