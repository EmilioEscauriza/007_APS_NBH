"""
real_space_2d_fft_comparison.py

Young's double slit in 2D: two parallel slabs separated by a gap.

The FFT shows sinc fringes (from slab width) modulated by cosine
interference fringes (from the slab separation).  Giving the two slabs
different phases (representing twin domains with different lattice
displacements) shifts the cosine modulation, making the pattern asymmetric.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

N = 256
slab_width = 4          # thickness of each slab [px]
separation = 8         # centre-to-centre distance [px]
slab_extent = 16        # horizontal width of each slab [px]
delta_phi = np.pi / 3   # phase offset of slab B (0 = symmetric)

cx, cy = N // 2, N // 2
x_lo = cx - slab_extent // 2
x_hi = cx + slab_extent // 2

img = np.zeros((N, N), dtype=complex)

# Slab A (below centre)
y_a = cy - separation // 2
img[y_a - slab_width // 2 : y_a + slab_width // 2, x_lo:x_hi] = 1.0

# Slab B (above centre) with phase offset
y_b = cy + separation // 2
img[y_b - slab_width // 2 : y_b + slab_width // 2, x_lo:x_hi] = np.exp(1j * delta_phi)

amp = fftshift(fft2(img))
power = np.abs(amp) ** 2

# ---- Plot ----
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

# Top-left: amplitude
ax_amp = fig.add_subplot(gs[0, 0])
ax_amp.imshow(np.abs(img), origin="lower", cmap="gray", vmin=0, vmax=1)
ax_amp.set_title("Amplitude  |ψ(r)|")
ax_amp.set_xlabel("x [px]")
ax_amp.set_ylabel("y [px]")

# Top-middle: phase
ax_ph = fig.add_subplot(gs[0, 1])
phase_display = np.angle(img)
phase_display[np.abs(img) < 1e-12] = np.nan
ax_ph.imshow(phase_display, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
ax_ph.set_title(f"Phase  (Δφ = {delta_phi/np.pi:.2f}π)")
ax_ph.set_xlabel("x [px]")

# Top-right: full FFT
ax_full = fig.add_subplot(gs[0, 2])
ax_full.imshow(np.log1p(power), origin="lower", cmap="inferno")
ax_full.set_title(r"FFT  $\log(1+|F|^2)$")
ax_full.set_xlabel(r"$q_x$")
ax_full.set_ylabel(r"$q_y$")

# Bottom-left: zoomed FFT
ax_zoom = fig.add_subplot(gs[1, 0:2])
w = 80
zoomed = np.log1p(power[cy - w:cy + w, cx - w:cx + w])
ax_zoom.imshow(zoomed, origin="lower", cmap="inferno",
               extent=[-w, w, -w, w])
ax_zoom.set_title(r"Zoomed FFT  $\log(1+|F|^2)$")
ax_zoom.set_xlabel(r"$\Delta q_x$ [px]")
ax_zoom.set_ylabel(r"$\Delta q_y$ [px]")
ax_zoom.axhline(0, color="white", lw=0.5, ls="--", alpha=0.5)

# Bottom-right: vertical linecut through q_x = 0
ax_cut = fig.add_subplot(gs[1, 2])
linecut = power[:, cx]
qy = np.arange(N) - cy
ax_cut.semilogy(qy, linecut + 1, "C0", lw=1.0)
ax_cut.set_xlabel(r"$\Delta q_y$ [px]")
ax_cut.set_ylabel(r"$|F|^2 + 1$")
ax_cut.set_title("Vertical linecut ($q_x = 0$)")
ax_cut.axvline(0, color="gray", lw=0.5, ls="--")
ax_cut.set_xlim(-80, 80)

fig.suptitle(
    f"Double slit — width={slab_width}, sep={separation}, "
    f"extent={slab_extent}, Δφ={delta_phi/np.pi:.2f}π  ({N}×{N})",
    fontsize=12,
)
plt.show()
