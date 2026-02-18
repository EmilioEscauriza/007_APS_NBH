# TTC Anti-Diagonal Interpretation

*Tuesday, January 20, 2026*

---

## What an anti-diagonal lineout measures

Along the anti-diagonal through $t_1 = t_2 = T$:

$$(t_1, t_2) = (T+\Delta, T-\Delta) \quad\Rightarrow\quad \tau = -2\Delta$$

So an anti-diagonal lineout is effectively $C(T,\tau)$ at fixed experimental time $T$. This is the cleanest possible measurement of intrinsic temporal structure, uncontaminated by aging or drift.

---

## What oscillations cannot be

An oscillation along the anti-diagonal **cannot** be due to:

1. **Aging**: Lives in $T$. You have fixed $T$.

2. **Slow drift** (temperature, beam, alignment): Breaks symmetry between $t_1$ and $t_2$. Anti-diagonal is explicitly symmetric.

3. **Simple diffusive relaxation**: Produces monotonic decay in $\tau$, never oscillations.

**Conclusion**: If oscillations are real, they are not a trivial artifact of non-stationarity.

---

## What oscillations do mean physically

An oscillation in $C(T,\tau)$ implies the system "remembers" past configurations in a phase-coherent way. This narrows interpretation enormously.

---

## Physical interpretations (ordered by plausibility)

### 1. Periodic or quasi-periodic internal dynamics

**Physical picture**: Some internal degree of freedom undergoes recurrent motion:
- Hopping back and forth
- Switching between configurations
- Elastic relaxation followed by recovery
- Collective modes in confined regions

If the system partially revisits similar configurations after time $\tau_0$:

$$C(T,\tau) \sim \cos(\omega \tau)\, e^{-\tau/\tau_c}$$

with oscillation period $2\pi/\omega$ and damping time $\tau_c$.

**Concrete examples**:
- Grain-boundary sliding that repeats
- Domain wall oscillations pinned between defects
- Strain relaxation and rebound
- Ionic rearrangements in a constrained cage

This is **not normal diffusion**. It is underdamped or quasi-reversible motion.

### 2. Periodic driving hidden inside the experiment

Sometimes oscillations reflect an external clock invisible in the time trace but visible in correlations:
- Cryostat temperature regulation
- Pressure regulation
- Beamline feedback loops
- Mechanical vibrations

**Why this shows up in anti-diagonal**: The TTC integrates over pairs of times, so even weak periodic forcing becomes coherent in $C(T,\tau)$.

**Rule of thumb**:
- Constant period over many $T$ → suspicious of instrumentation
- Slowly drifting or appearing only at specific $T$ → likely intrinsic

### 3. Intermittent switching between metastable states

Extremely relevant to ionic conductors and complex solids. Imagine:
- Two or more structural microstates
- Switching is not random but has preferred timescales
- System "rings" after a transition

Then:
- Anti-diagonal lineout shows oscillations
- Diagonal-parallel cuts show checkerboard or blocky TTC textures
- Different masks show oscillations at different phases or amplitudes

This matches mask-selective oscillations.

### 4. Collective modes in Bragg XPCS

**Important nuance**: In Bragg geometry, you measure phase-sensitive strain or displacement fields, not particle positions.

That means:
- Small coherent shifts can strongly affect speckle
- Elastic modes can produce oscillatory correlations
- Even Å-scale reversible motion is visible

So oscillations do **not** require large atomic motion.

---

## Diagnostic tests

### A. Compare different masks

- Same frequency everywhere → global mode or external driver
- Mask-dependent phase or frequency → local dynamics

### B. Track frequency vs $T$

- Constant → instrumental or steady collective mode
- Slowly changing → evolving microstructure

### C. Check symmetry

True anti-diagonal oscillations must be:

$$C(T,+\tau) = C(T,-\tau)$$

If not, you're mixing in aging or drift.

### D. Compare with diagonal-parallel cuts

- Oscillation only in anti-diagonal → intrinsic dynamics
- Oscillation also in diagonal-parallel → aging + recurrence

---

## Key conceptual takeaways

1. An oscillation along the anti-diagonal means the system is **not simply relaxing** — it is dynamically structured in time. That is a strong statement.

2. In many XPCS papers, seeing this would already justify:
   - "Non-diffusive dynamics"
   - "Reversible or quasi-elastic rearrangements"
   - "Collective modes"
   - "Periodic/intermittent microscopic processes"

3. Anti-diagonal oscillations:
   - Are immune to most trivial artifacts
   - Are rarely shown
   - Strongly support claims of non-equilibrium or complex dynamics

Handled carefully, this is a central result, not a side observation.

---

## Overall conclusion

Oscillations observed along the anti-diagonal of a TTC map indicate that the system does not undergo simple relaxational dynamics but instead exhibits temporally structured, partially reversible microscopic processes. This constitutes strong evidence for non-diffusive, collective, or intermittently driven dynamics in the probed material.
