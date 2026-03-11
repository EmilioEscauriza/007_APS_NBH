# Single-Frequency TTC Model (ω and 2ω)

*Wednesday, January 28, 2026*

---

## Overview

A clean physical picture separates what fluctuates, where it fluctuates, and how those fluctuations are coupled in time. The model uses two frequencies: **ω** (base) and **2ω** (second harmonic).

---

## Model structure

$$C_2(t_1, t_2) = C_0 + A\cos(2\omega v) + m\cos(2\omega u + \phi)\cos(\omega v)$$

where:
- $u = (t_1 + t_2)/2$ (mean time)
- $v = t_2 - t_1$ (lag time)

Linked intensity-like trace (for intuition): $I(t) = C_0 + a_I\cos(2\omega t) + m_I\cos(\omega t + \phi)$.

---

## Term-by-term interpretation

### 1. $A\cos(2\omega v)$: Stationary oscillatory dynamics (2ω in delay)

Depends only on lag time $v$. Physically:

- Stationary, time-translation–invariant process
- Oscillatory mode with period $T_{2\omega} = \pi/\omega$ in the delay direction
- Coherent relaxation, hopping, or vibrational process that is always present

In XPCS language: subset of scatterers undergoes periodic/quasi-periodic dynamics that do not age or drift during measurement. This is the "safe" term, behaving like $g_2$-based intuition expects. The 2ω gives finer stripe spacing in the TTC.

### 2. $m\cos(2\omega u + \phi)$: Non-stationarity or slow modulation

Depends on mean time $u$. Means:

- Dynamics are not the same at early and late times
- System is modulated at 2ω in mean time (period $T = \pi/\omega$ in $u$)

Physically represents:
- Environmental or structural evolution at the second harmonic
- Periodic activation of dynamics (e.g., stress build-up and release)
- Switching between dynamical states

In Bragg XPCS context: grain-boundary rearrangements, intermittent defect motion, phase-front breathing, or periodic strain accumulation. This term alone violates stationarity, hence the TTC structure.

### 3. $\cos(\omega v)$: Base-frequency lag-time structure

In the product term, the lag-time factor is at **ω** (slower than the 2ω stripes):

- Lag-time oscillation has period $T_\omega = 2\pi/\omega$
- Complements the 2ω term: two frequencies in the delay direction

Physically suggests:
- Base mode in the correlation
- First harmonic (ω) and second harmonic (2ω) both present—e.g. anharmonic or intensity-quadratic response

Natural for: second-harmonic generation in the correlation, anharmonic motion, or detection sensitive to both ω and 2ω.

### 4. Product structure: $\cos(2\omega u + \phi)\cos(\omega v)$

Means lag-time correlations at ω are gated by mean-time modulation at 2ω. Produces:

- Diamond or chequerboard TTC patterns
- Oscillations localized in $u$
- Masks where oscillations appear and disappear

Matches observation that only specific masks around the Bragg peak show oscillations, the diagonal is not always strongest, and TTC shows repeating but localized structure.

---

## Why the two frequencies (ω and 2ω) matter

### Physical meaning

The model has **ω** (base) and **2ω** (second harmonic):

- Stripes and mean-time modulation use **2ω** (faster).
- The product term also has **ω** in the delay direction (slower).

So the correlation carries both a fundamental and its first harmonic—e.g. anharmonic motion or intensity ∝ displacement².

### Why 2ω appears

- **Intensity ∝ displacement²:** If the signal is quadratic in displacement $x(t) \sim \cos(\omega t)$, then $x^2 \sim \cos(2\omega t) + \text{const}$, so 2ω appears.
- **First harmonic:** Nonlinear or anharmonic dynamics excite 2ω alongside ω.

Common in: anharmonic oscillators, intensity-sensitive detection, nonlinear coupling.

### Relation to the product term

Model structure:

$$C_0 + A\cos(2\omega v) + m\cos(2\omega u + \phi)\cos(\omega v)$$

Interpretation:

- Stripes: 2ω in delay $v$.
- Modulated term: 2ω in mean time $u$, ω in delay $v$—so the same base ω drives the slower lag structure, and 2ω drives the faster stripes and modulation.

The linked trace $I(t) = C_0 + a_I\cos(2\omega t) + m_I\cos(\omega t + \phi)$ reflects both frequencies in a single-time view.

---

## Physical interpretation (one sentence)

A periodically active nanoscale process is modulated at 2ω in mean time; the correlation carries both ω and 2ω in the delay direction, producing non-stationary, two-frequency structure in the TTC.

---

## Why this is compelling for Bragg XPCS

At a Bragg peak:

- Sensitive to phase and strain, not just density
- Small reversible displacements can dominate the signal
- Collective motion of ordered regions naturally produces oscillatory TTC features
- Intensity can depend quadratically on displacement, giving 2ω from ω motion

The model is the minimal mathematical structure capturing:
- Stationary dynamics (2ω in $v$)
- Non-stationarity (2ω in $u$)
- Mode coupling (product term)
- Two-frequency (ω and 2ω) temporal structure

---

## Experimental implications

If the two-frequency (ω and 2ω) structure is real (not a fitting artifact), then:

- Dynamics have a measurable second harmonic (anharmonic or quadratic response)
- Both base and first harmonic appear in the correlation
- Modulated term ties mean-time (2ω) and lag-time (ω and 2ω) together

This also explains:
- Why only some masks show oscillations (modulation in $u$)
- Why TTC diagonals aren't always maximal
- Why $g_2$ alone hides the physics

---

## One-sentence takeaway

The model uses ω (base) and 2ω (second harmonic): stripes and mean-time modulation at 2ω, with ω in the product’s delay factor and in the linked trace $I(t)$, consistent with anharmonic or intensity-quadratic dynamics and non-stationary TTC structure.
