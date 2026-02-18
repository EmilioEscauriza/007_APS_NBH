# Single-Frequency TTC Model

*Wednesday, January 28, 2026*

---

## Overview

A clean physical picture separates what fluctuates, where it fluctuates, and how those fluctuations are coupled in time.

---

## Model structure

$$C_2(t_1, t_2) = A\cos(\omega v) + m\cos(\omega u + \phi)\cos\!\left(\tfrac{\omega}{2}v\right)$$

where:
- $u = (t_1 + t_2)/2$ (mean time)
- $v = t_2 - t_1$ (lag time)

---

## Term-by-term interpretation

### 1. $A\cos(\omega v)$: Stationary oscillatory dynamics

Depends only on lag time $v$. Physically:

- Stationary, time-translation–invariant process
- Well-defined oscillatory mode with period $T = 2\pi/\omega$
- Coherent relaxation, hopping, or vibrational process that is always present

In XPCS language: subset of scatterers undergoes periodic/quasi-periodic dynamics that do not age or drift during measurement. This is the "safe" term, behaving like $g_2$-based intuition expects.

### 2. $m\cos(\omega u + \phi)$: Non-stationarity or slow modulation

Depends on mean time $u$. Means:

- Dynamics are not the same at early and late times
- System is slowly modulated, intermittently activated, or drifting

Physically represents:
- Slow environmental or structural evolution
- Periodic activation of dynamics (e.g., stress build-up and release)
- Switching between dynamical states

In Bragg XPCS context: grain-boundary rearrangements, intermittent defect motion, phase-front breathing, or periodic strain accumulation. This term alone violates stationarity, hence the TTC structure.

### 3. $\cos(\tfrac{\omega}{2} v)$: Subharmonic lag-time structure

Compared to the main $A$-term:

- Lag-time oscillation is slower by a factor of 2
- System "remembers" correlations over longer lag times

Physically suggests:
- Two-step or paired process
- Correlations requiring two events to decorrelate fully
- Alternating forward–backward motion, stick–slip, or reversible rearrangements

Natural for: periodic grain-boundary diffusion, back-and-forth defect migration, elastic loading/unloading cycles.

### 4. Product structure: $\cos(\omega u)\cos(\tfrac{\omega}{2} v)$

Means lag-time correlations only appear strongly during certain global-time windows. Produces:

- Diamond or chequerboard TTC patterns
- Oscillations localized in $u$
- Masks where oscillations appear and disappear

Matches observation that only specific masks around the Bragg peak show oscillations, the diagonal is not always strongest, and TTC shows repeating but localized structure.

---

## Why the half-frequency term matters

### Physical meaning

The term $\cos(\tfrac{\omega}{2} v)$ means:

- Correlation repeats only after **twice** the period of the underlying oscillation
- System needs two cycles before it "looks the same again" in lag time
- Classic period doubling in correlations, not necessarily in the motion itself

### Why correlations can have half the frequency

For microscopic displacement $x(t) = x_0 \cos(\omega t)$, scattering phase depends on:

$$\Delta \phi \sim q [x(t_2) - x(t_1)]$$

After half a cycle, displacement reverses sign. Structure may be in a different configuration but not fully decorrelated. Only after a full forward–backward cycle does the phase difference repeat. So:

$$\omega_\text{corr} = \tfrac{1}{2}\omega_\text{motion}$$

Common in: reversible motion, elastic deformation, back-and-forth defect hopping.

### Two-state or back-and-forth dynamics

Half-frequency almost always means the system has two equivalent configurations:

- Grain boundary: left → right → left
- Defect: A → B → A
- Strain: loads → unloads → loads

Microscopic motion has frequency $\omega$, but configuration repeats every two steps, so correlations repeat at $\omega/2$. This explains diamonds/chequerboards, alternating bright/dark lobes, and strong anti-diagonal structure in TTC.

### Why this matters for Bragg XPCS

At a Bragg peak:

- Intensity is sensitive to **phase**, not just magnitude
- Opposite displacements can produce similar intensities
- Sign changes in displacement don't immediately decorrelate the speckle

So Bragg XPCS is especially prone to half-frequency effects. In diffuse scattering you might lose this; in Bragg it survives.

### Why half-frequency appears only in the modulated term

Model structure:

$$A\cos(\omega v) + m\cos(\omega u)\cos\!\left(\tfrac{\omega}{2}v\right)$$

Interpretation:

- Stationary background dynamics decorrelate every cycle → full frequency
- Intermittent/gated dynamics require two cycles → half frequency

These are **not the same physical process**. This is a huge experimental clue.

---

## Physical interpretation (one sentence)

A periodically active nanoscale process (grain boundary, defect cluster, strain front) undergoes coherent oscillatory motion, but only during certain phases of a slower cycle, producing non-stationary, subharmonic correlations in the TTC.

---

## Why this is compelling for Bragg XPCS

At a Bragg peak:

- Sensitive to phase and strain, not just density
- Small reversible displacements can dominate the signal
- Collective motion of ordered regions naturally produces oscillatory TTC features

The model is the minimal mathematical structure capturing:
- Stationary dynamics
- Non-stationarity
- Mode coupling
- Subharmonic temporal structure

---

## Experimental implications

If the half-frequency is real (not a fitting artifact), then:

- Dynamics are reversible or near-reversible
- Seeing collective motion, not random diffusion
- Grain boundaries or strain fields far more likely than ionic hopping
- System is not ergodic on the measurement timescale

This also explains:
- Why only some masks show oscillations
- Why TTC diagonals aren't always maximal
- Why $g_2$ alone hides the physics

---

## One-sentence takeaway

A half-frequency in $v$ means the system must complete a full forward-and-back cycle before its microscopic configuration truly repeats, which is the hallmark of reversible, collective dynamics such as elastic or grain-boundary motion.
