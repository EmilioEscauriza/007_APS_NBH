# Raw Mask Time Trace and FFT Inspection

*Friday, January 23, 2026*

---

## Methodology

### Processing pipeline

1. **Raw detector data**: Start from frames $D(t, x, y)$ where $t = 0, 1, \ldots, T-1$ is the frame index and $(x,y)$ are detector pixel coordinates.

2. **ROI masking**: Construct binary mask $M_k(x,y)$ for ROI $k$. Only pixels where $M_k=1$ are used.

3. **Frame-wise ROI intensity**: For each frame, compute mean intensity:

   $$I_k(t) = \frac{1}{N_k}\sum_{x,y} M_k(x,y) \, D(t,x,y)$$

   where $N_k = \sum_{x,y} M_k(x,y)$ is the number of pixels in the ROI. This produces a scalar time series $I_k(t)$.

4. **Mean removal (DC subtraction)**: Remove long-time mean:

   $$I_k'(t) = I_k(t) - \frac{1}{T}\sum_{t'} I_k(t')$$

   This centers the trace around zero, removes absolute intensity, and isolates fluctuations.

5. **Optional linear detrending**: Fit and subtract a linear model to remove slow intensity ramps, beam decay, thermal drift, and scan envelope effects. What remains is stationary fluctuation content.

6. **Windowing**: Apply Hann window before FFT:

   $$w(t) = \frac{1}{2}\left[1 - \cos\left(\frac{2\pi t}{T-1}\right)\right]$$

   This suppresses spectral leakage from finite time boundaries. Endpoints go smoothly to zero, edge discontinuities are removed, and power from a real frequency stays localized.

7. **Discrete Fourier Transform**: Compute real FFT:

   $$\hat{I}\_{k}(f) = \sum\_{t=0}^{T-1} I\_{k}^{\prime}(t) \, w(t) \, e^{-2\pi \mathrm{i} f t \Delta t}$$

   where $\Delta t$ is the frame time (seconds) and frequencies are $f = 0, 1/(T\Delta t), 2/(T\Delta t), \ldots$.

8. **Power spectrum**: Define as $P_k(f) = |\hat{I}\_k(f)|^2$ with DC component ($f=0$) explicitly ignored. Plot in log scale.

9. **Peak frequency selection**: Within a physically motivated band, identify the dominant oscillatory component $f_\text{peak}$.

10. **Period extraction**: Oscillation period is $T_\text{peak} = 1 / f_\text{peak}$. Compare across signal ROI, control ROI, different diagonal start times, and TTC lineouts.

---

## Results for A073

### Signal ROI (mask 145) vs Control ROI (mask 86, 144, 176, 0, 3)

**Time traces:**

- **Signal**: Huge amplitude relative to control (orders of magnitude larger). Very clear low-frequency structure riding under noise. Broad "U-shaped" envelope with slower modulations. Fluctuations are **not symmetric noise around zero** — this is not random speckle noise.

- **Control**: Much smaller amplitude. Symmetric, stationary-looking fluctuations. No obvious slow modulation, just noise + envelope from beam decay. Looks like photon counting noise, slow beam intensity decay, no internal dynamics.

**FFTs:**

- **Signal**: Strong enhancement at very low frequency. Power rises sharply as $f \to 0$. Clear excess spectral weight below ~0.01 Hz. This is what a slow, quasi-periodic modulation produces in FFT space. Even without a razor-sharp peak, this is not white noise.

- **Control**: Flat-ish spectrum on log scale. No low-frequency enhancement. No excess near the signal's low-frequency region. Textbook "noise + counting statistics + slow drift removed".

### Critical comparison

**There is no shared spectral feature between signal and control.**

This immediately rules out:
- Beam current oscillations
- Monochromator feedback
- Shutter timing artifacts
- Detector electronics oscillations
- Temperature controller cycling (unless highly spatially selective, which is implausible)

If this were instrumental, both FFTs would light up in the same place. They don't.

---

## Interpretation logic

- Raw images show absolute photon counts
- Mean-subtracted traces show fluctuations
- FFT peaks indicate coherent temporal structure
- Control ROI should show broadband noise, no persistent peak
- Signal ROI shows a reproducible peak consistent with TTC oscillations
- Mean removal and detrending cannot create oscillations — they only remove low-frequency content

**Conclusion**: Signal and control are qualitatively different already at the time-trace level.
