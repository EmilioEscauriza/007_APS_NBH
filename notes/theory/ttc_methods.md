# Comparison of four TTC methods

*Tuesday, February 3, 2026*

---

## ChatGPT's twotime-like method

### 1. What object is being built first: $I(t,p)$

From the raw detector frames you construct an intensity matrix $I(t,p)$ where:

- **t** = frame index (time)
- **p** = pixel index inside the selected ROI (mask)

This comes from:

```text
I, frame_idxs = extract_roi_intensity_matrix(...)
```

So at this point:

- **Shape**: `I.shape == (T, P)`
- **T** = number of selected frames
- **P** = number of pixels in the ROI

This is exactly the data structure that `twotime.py` operates on internally.

### 2. Per-pixel normalization (“smooth data” step)

Next you apply:

```text
I_smooth = _smooth_like_twotime(I)
```

**Mathematically:** for each pixel $p$, compute its time-mean, then normalize:

$$\bar{I}(p) = \frac{1}{T}\sum_{t=1}^{T} I(t,p)$$

Guard against zeros / negatives: $\bar{I}(p) \leftarrow 1$ if $\bar{I}(p) \le 0$.

Then each pixel's time trace is normalized by its own mean:

$$I_\text{smooth}(t,p) = \frac{I(t,p)}{\bar{I}(p)}$$

**Key consequences:**

- Removes static pixel-to-pixel intensity differences
- Preserves temporal correlations
- Makes all pixels contribute equally to the TTC

This step is crucial and often overlooked.

### 3. Core two-time correlation (twotime math)

Computed by:

```text
C_raw_ut = _compute_ttc_like_twotime(I_smooth)
```

This mirrors `twotime.py::calc_normal_twotime`.

#### 3.1 Pixel-averaged cross product

Pixel-averaged intensity product:

$$M(t_1, t_2) = \sum_{p=1}^{P} I_\text{smooth}(t_1,p)\, I_\text{smooth}(t_2,p)$$

In matrix form:

$$M = I_\text{smooth} \, I_\text{smooth}^\top \quad \text{or} \quad \text{cross} = (I @ I^\top) / P$$

#### 3.2 Frame-wise normalization

Total ROI intensity per frame:

$$S(t) = \sum_{p=1}^{P} I_\text{smooth}(t,p)$$

Guard: $S(t) \leftarrow 1$ if $S(t) \le 0$.

Normalization factor:

$$n(t) = \frac{1}{S(t)}$$

This enforces intensity normalization per frame, not per pixel.

#### 3.3 Final TTC definition (twotime convention)

$$C_2(t_1,t_2) = M(t_1,t_2)\, n(t_1)\, n(t_2)\, P$$

Equivalently:

$$C_2 = \left(I_\text{smooth} I_\text{smooth}^\top\right) \odot \left(n n^\top\right) \cdot P$$

where $\odot$ is elementwise multiplication.

In code:

```text
matmul_prod = ts @ ts.T
c2 = matmul_prod * norm_factor[:, None] * norm_factor[None, :] * npix
```

### 4. Upper-triangle storage and symmetrization

`twotime.py` only computes $C_2(t_1,t_2)$ for $t_2 \ge t_1$, then:

$$C(t_1,t_2) = C(t_2,t_1)$$

Stored as `triu(C_2)`. Your `_symmetrize_upper_triangle()` reconstructs the full symmetric TTC for plotting.

### 5. What this TTC actually measures

This TTC is **not** $\langle I(t_1) I(t_2) \rangle$ and is **not** a standard Pearson correlation.

It is a **normalized pixel-ensemble intensity overlap**:

- Sensitive to reproducibility of the speckle pattern
- Insensitive to static intensity offsets
- Sensitive to slow, collective rearrangements
- Exactly what Bragg-XPCS needs

### 6. Why the comparison (RAW − PROCESSED) is meaningful

Both raw and processed TTCs use the same pixel set, normalization, and frame indexing. Any difference in RAW − PROCESSED must come from:

- Preprocessing differences
- Masking differences
- Numerical smoothing / batching
- Or bugs (which this comparison is designed to catch)

So this is a proper validation tool, not just a visual check.

---

## Twotime.py APS 8-ID-E method

**Inputs:**

- A stack of raw detector frames: $D(t, y, x)$
- A ROI definition (mask) selecting pixels $p = 1 \ldots P$

**ROI intensity matrix:**

$$I(t,p) \quad \text{shape } (T, P)$$

- $t$ = frame index (time)
- $p$ = pixel index within the ROI

### Step 1: Per-pixel smoothing / normalization

For each pixel $p$, time-mean:

$$\bar{I}(p) = \frac{1}{T}\sum_{t=1}^{T} I(t,p)$$

Guard: $\bar{I}(p) \leftarrow 1$ if $\bar{I}(p) \le 0$.

Normalize:

$$I_\text{smooth}(t,p) = \frac{I(t,p)}{\bar{I}(p)}$$

This step is optional in twotime.py (depends on flags); when enabled it happens before computing the TTC.

### Step 2: Frame-wise normalization factor

$$S(t) = \sum_{p=1}^{P} I_\text{smooth}(t,p)$$

Guard: $S(t) \leftarrow 1$ if $S(t) \le 0$.

$$n(t) = \frac{1}{S(t)}$$

($n(t)$ is a length-$T$ vector.)

### Step 3: Pixel-averaged cross-product matrix

$$M(t_1, t_2) = \sum_{p=1}^{P} I_\text{smooth}(t_1,p)\, I_\text{smooth}(t_2,p)$$

Matrix form: $M = I_\text{smooth} I_\text{smooth}^\top$ (a $T \times T$ matrix).

### Step 4: Apply normalization and pixel-count scaling

$$C_2(t_1,t_2) = M(t_1,t_2)\, n(t_1)\, n(t_2)\, P$$

$$C_2 = \left(I_\text{smooth} I_\text{smooth}^\top\right) \odot \left(n n^\top\right) \cdot P$$

### Step 5: Keep only the upper triangle

Store only $C_2(t_1,t_2)$ for $t_2 \ge t_1$, i.e. $C_2 \leftarrow \mathrm{triu}(C_2)$.

---

## Summary: implementation steps (one ROI)

1. Build $I(t,p)$ from raw frames using the ROI pixel map.
2. (Optional) Normalize each pixel column by its own time mean.
3. Compute frame sums $S(t)$ and $n(t)=1/S(t)$.
4. Compute $M = I I^\top$.
5. Compute $C_2 = M \cdot n(t_1)n(t_2) \cdot P$.
6. Store only $\mathrm{triu}(C_2)$.

**In your code:**

- `_smooth_like_twotime()` → Step 1
- `_compute_ttc_like_twotime()` → Steps 2–5

---

## Comparison table

Notation: **P** = pixels in ROI.

| Method        | What it averages over | Pre-processing              | Normalization                    | Sensitive to                                                                 |
|---------------|----------------------|-----------------------------|----------------------------------|-------------------------------------------------------------------------------|
| **G-TTC**     | Pixels in ROI        | None (raw intensities)      | Per-frame variance normalization | Fluctuations around the mean, contrast changes                                |
| **Corr-TTC**  | Pixels in ROI        | None (raw intensities)      | Mean-intensity normalization only | Intensity correlations including slow drifts                                  |
| **Twotime.py TTC** | Pixels in ROI   | Per-pixel temporal smoothing | Per-pixel mean + per-frame sum   | Relative pixel-wise fluctuations, suppresses static structure                |
| **CGPT TTC**  | Pixels in ROI        | Per-pixel temporal smoothing | Same as twotime.py               | Same as twotime.py                                                            |
