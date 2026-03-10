"""
bragg_forward_model.py

BCDI forward model: compute the 3D coherent diffraction intensity around
a single Bragg peak via FFT.

Physics (kinematic approximation):
    1. Define a 3D real-space object on an N×N×N grid (voxel size Δr).
    2. Support ρ(r): particle shape (sphere, cube, or faceted).
    3. Displacement u(r) = ε·r  (uniform strain tensor ε, 3×3 symmetric).
    4. Complex object  ψ(r) = ρ(r) · exp(i G_hkl · u(r)).
    5. Diffraction intensity  I(Q) = |FFT[ψ]|².
    6. Optionally add Poisson counting noise.

    q-space resolution:  Δq = 2π / (N · Δr)
    q-range:  ±N·Δq/2  around the Bragg peak.

Default lattice parameters are for orthorhombic NaBH₄ at 12.4 keV.
"""

from __future__ import annotations

import argparse
import numpy as np
from dataclasses import dataclass, field
from numpy.fft import fftn, fftshift, fftfreq
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ---------------------------------------------------------------------------
#  Physical constants
# ---------------------------------------------------------------------------
HC_KEV_A = 12.3984193  # hc [keV·Å]


# ---------------------------------------------------------------------------
#  Reciprocal-lattice helper (standard crystallography, no external deps)
# ---------------------------------------------------------------------------
def reciprocal_lattice_vectors(
    a: float, b: float, c: float,
    alpha_deg: float, beta_deg: float, gamma_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reciprocal-lattice basis vectors (b1, b2, b3) in Å⁻¹.

    Direct-space basis vectors are constructed in the Cartesian frame
    with a1 along x and a2 in the x-y plane (standard convention).
    The reciprocal vectors satisfy  bi · aj = 2π δij.
    """
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sg = np.sin(gamma)

    a1 = a * np.array([1.0, 0.0, 0.0])
    a2 = b * np.array([cg, sg, 0.0])
    a3_x = c * cb
    a3_y = c * (ca - cb * cg) / sg
    a3_z = c * np.sqrt(1.0 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg) / sg
    a3 = np.array([a3_x, a3_y, a3_z])

    V = np.dot(a1, np.cross(a2, a3))
    b1 = 2.0 * np.pi * np.cross(a2, a3) / V
    b2 = 2.0 * np.pi * np.cross(a3, a1) / V
    b3 = 2.0 * np.pi * np.cross(a1, a2) / V
    return b1, b2, b3


# ---------------------------------------------------------------------------
#  Parameter dataclass
# ---------------------------------------------------------------------------
@dataclass
class BCDIForwardParams:
    """All tuneable parameters for the BCDI forward model."""

    # --- Beam ---
    energy_keV: float = 12.4

    # --- Crystal lattice (direct space, Å and degrees) ---
    a: float = 6.13   # NaBH₄ orthorhombic a [Å]
    b: float = 6.13   # b [Å]
    c: float = 6.13   # c [Å]
    alpha_deg: float = 90.0
    beta_deg: float = 90.0
    gamma_deg: float = 90.0

    # --- Reflection ---
    hkl: tuple[int, int, int] = (1, 1, 0)

    # --- Real-space grid ---
    grid_size: int = 128        # N for N×N×N
    voxel_size_nm: float = 5.0  # real-space voxel [nm]

    # --- Particle ---
    particle_shape: str = "sphere"   # "sphere", "cube", "faceted"
    particle_size_nm: float = 300.0  # characteristic diameter / side [nm]

    # --- Strain (Voigt: xx, yy, zz, yz, xz, xy) ---
    strain_voigt: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # --- Noise ---
    max_photons: float | None = None  # Poisson scaling; None = no noise

    # --- Derived ---
    wavelength_A: float = field(init=False)
    d_spacing_A: float = field(init=False)
    theta_B_deg: float = field(init=False)
    G_hkl: np.ndarray = field(init=False, repr=False)
    delta_q_inv_A: float = field(init=False)

    def __post_init__(self):
        self._recompute()

    # ---- helpers ----
    @property
    def strain_tensor(self) -> np.ndarray:
        """3×3 symmetric strain tensor from Voigt notation."""
        xx, yy, zz, yz, xz, xy = self.strain_voigt
        return np.array([
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ])

    def _recompute(self):
        self.wavelength_A = HC_KEV_A / self.energy_keV

        b1, b2, b3 = reciprocal_lattice_vectors(
            self.a, self.b, self.c,
            self.alpha_deg, self.beta_deg, self.gamma_deg,
        )
        h, k, l = self.hkl
        self.G_hkl = h * b1 + k * b2 + l * b3  # Å⁻¹

        G_mag = np.linalg.norm(self.G_hkl)
        self.d_spacing_A = 2.0 * np.pi / G_mag

        sin_theta = self.wavelength_A / (2.0 * self.d_spacing_A)
        if abs(sin_theta) > 1.0:
            raise ValueError(
                f"Bragg condition impossible: λ={self.wavelength_A:.4f} Å, "
                f"d={self.d_spacing_A:.4f} Å → sin(θ)={sin_theta:.4f}"
            )
        self.theta_B_deg = np.degrees(np.arcsin(sin_theta))

        voxel_A = self.voxel_size_nm * 10.0  # nm → Å
        self.delta_q_inv_A = 2.0 * np.pi / (self.grid_size * voxel_A)


# ---------------------------------------------------------------------------
#  Support generation
# ---------------------------------------------------------------------------
def make_support(params: BCDIForwardParams) -> np.ndarray:
    """Return a binary 3D support array (N, N, N)."""
    N = params.grid_size
    half = N // 2
    radius_vox = (params.particle_size_nm / 2.0) / params.voxel_size_nm

    zz, yy, xx = np.mgrid[-half:N - half, -half:N - half, -half:N - half]

    shape = params.particle_shape.lower()
    if shape == "sphere":
        dist = np.sqrt(xx**2 + yy**2 + zz**2)
        support = (dist <= radius_vox).astype(np.float64)
    elif shape == "cube":
        support = (
            (np.abs(xx) <= radius_vox)
            & (np.abs(yy) <= radius_vox)
            & (np.abs(zz) <= radius_vox)
        ).astype(np.float64)
    elif shape == "faceted":
        try:
            from bcdi.simulation.supportMaker import generatePlanesCuboid, MakePoly
            side = int(2 * radius_vox)
            side = max(side, 3)
            planes = generatePlanesCuboid(side, side, side)
            support = MakePoly((N, N, N), planes).astype(np.float64)
        except ImportError:
            dist = np.sqrt(xx**2 + yy**2 + zz**2)
            support = (dist <= radius_vox).astype(np.float64)
    else:
        raise ValueError(f"Unknown particle_shape '{params.particle_shape}'")

    return support


# ---------------------------------------------------------------------------
#  Forward model
# ---------------------------------------------------------------------------
def forward_model(
    params: BCDIForwardParams,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute 3D coherent diffraction intensity around a Bragg peak.

    Returns
    -------
    intensity : (N, N, N) real array
    q_axes    : (qz, qy, qx) 1-D arrays in Å⁻¹, centred on the Bragg peak
    """
    N = params.grid_size
    voxel_A = params.voxel_size_nm * 10.0  # nm → Å

    support = make_support(params)

    # Real-space coordinate grids (centred, in Å)
    coords_1d = (np.arange(N) - N // 2) * voxel_A
    rz, ry, rx = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")

    # Displacement u(r) = ε · r  →  phase = G · u = G · ε · r
    eps = params.strain_tensor
    G = params.G_hkl
    # G·ε is a 1×3 vector; dot with r gives scalar phase per voxel
    Geps = G @ eps  # shape (3,)
    phase = Geps[0] * rx + Geps[1] * ry + Geps[2] * rz

    # Complex object
    psi = support * np.exp(1j * phase)

    # FFT → diffraction amplitude
    amp = fftshift(fftn(psi))
    intensity = np.abs(amp) ** 2

    # Poisson noise
    if params.max_photons is not None and params.max_photons > 0:
        scale = params.max_photons / intensity.max()
        intensity = np.random.poisson(intensity * scale).astype(np.float64)

    # q-axes (deviation from Bragg peak)
    q_1d = fftshift(fftfreq(N, d=voxel_A)) * 2.0 * np.pi  # Å⁻¹
    return intensity, (q_1d, q_1d, q_1d)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def plot_forward_model(
    intensity: np.ndarray,
    q_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    params: BCDIForwardParams,
    log_scale: bool = True,
) -> plt.Figure:
    """Plot central 2-D slices and a radial profile of the 3-D pattern."""
    qz, qy, qx = q_axes
    N = intensity.shape[0]
    mid = N // 2

    data = np.log1p(intensity) if log_scale else intensity
    label = "log(1 + I)" if log_scale else "I"

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # qx-qy slice
    ax = axes[0, 0]
    extent = [qx[0], qx[-1], qy[0], qy[-1]]
    im = ax.imshow(
        data[mid, :, :], origin="lower", extent=extent, aspect="equal", cmap="magma",
    )
    ax.set_xlabel(r"$\Delta q_x$ [$\AA^{-1}$]")
    ax.set_ylabel(r"$\Delta q_y$ [$\AA^{-1}$]")
    ax.set_title(f"Central $q_z$ slice  ({label})")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # qx-qz slice
    ax = axes[0, 1]
    extent = [qx[0], qx[-1], qz[0], qz[-1]]
    im = ax.imshow(
        data[:, mid, :], origin="lower", extent=extent, aspect="equal", cmap="magma",
    )
    ax.set_xlabel(r"$\Delta q_x$ [$\AA^{-1}$]")
    ax.set_ylabel(r"$\Delta q_z$ [$\AA^{-1}$]")
    ax.set_title(f"Central $q_y$ slice  ({label})")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # qy-qz slice
    ax = axes[1, 0]
    extent = [qy[0], qy[-1], qz[0], qz[-1]]
    im = ax.imshow(
        data[:, :, mid], origin="lower", extent=extent, aspect="equal", cmap="magma",
    )
    ax.set_xlabel(r"$\Delta q_y$ [$\AA^{-1}$]")
    ax.set_ylabel(r"$\Delta q_z$ [$\AA^{-1}$]")
    ax.set_title(f"Central $q_x$ slice  ({label})")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Radial profile
    ax = axes[1, 1]
    qr_3d = np.sqrt(
        qx[np.newaxis, np.newaxis, :] ** 2
        + qy[np.newaxis, :, np.newaxis] ** 2
        + qz[:, np.newaxis, np.newaxis] ** 2,
    )
    n_bins = N // 2
    qr_max = qr_3d.max()
    bin_edges = np.linspace(0, qr_max, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    digitised = np.digitize(qr_3d.ravel(), bin_edges) - 1
    flat = intensity.ravel()
    radial = np.zeros(n_bins)
    for i in range(n_bins):
        mask = digitised == i
        if mask.any():
            radial[i] = flat[mask].mean()

    ax.semilogy(bin_centres, radial + 1, "C0", lw=1.2)
    ax.set_xlabel(r"$|\Delta q|$ [$\AA^{-1}$]")
    ax.set_ylabel(r"$\langle I \rangle$ (radial)")
    ax.set_title("Radial average")

    h, k, l = params.hkl
    fig.suptitle(
        f"BCDI forward model — ({h}{k}{l}), "
        f"E={params.energy_keV:.1f} keV, "
        f"2θ={2 * params.theta_B_deg:.2f}°, "
        f"N={params.grid_size}, Δr={params.voxel_size_nm:.1f} nm",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
#  Interactive mode
# ---------------------------------------------------------------------------
def interactive_plot(params: BCDIForwardParams | None = None):
    """Launch an interactive figure with sliders for particle size
    and diagonal strain components."""
    if params is None:
        params = BCDIForwardParams(grid_size=64)

    fig = plt.figure(figsize=(13, 8))
    fig.subplots_adjust(left=0.07, right=0.56, bottom=0.08, top=0.92, hspace=0.35, wspace=0.35)

    ax_xy = fig.add_subplot(2, 2, 1)
    ax_xz = fig.add_subplot(2, 2, 2)
    ax_yz = fig.add_subplot(2, 2, 3)
    ax_rad = fig.add_subplot(2, 2, 4)
    plot_axes = [ax_xy, ax_xz, ax_yz, ax_rad]
    colorbars: list = []

    def draw():
        for cb in colorbars:
            cb.remove()
        colorbars.clear()

        try:
            params._recompute()
            intensity, (qz, qy, qx) = forward_model(params)
        except ValueError:
            return

        N = intensity.shape[0]
        mid = N // 2
        data = np.log1p(intensity)

        for ax in plot_axes:
            ax.cla()

        extent_xy = [qx[0], qx[-1], qy[0], qy[-1]]
        im = ax_xy.imshow(data[mid, :, :], origin="lower", extent=extent_xy, aspect="equal", cmap="magma")
        ax_xy.set_xlabel(r"$\Delta q_x$")
        ax_xy.set_ylabel(r"$\Delta q_y$")
        ax_xy.set_title("$q_z = 0$ slice")
        colorbars.append(fig.colorbar(im, ax=ax_xy, shrink=0.75))

        extent_xz = [qx[0], qx[-1], qz[0], qz[-1]]
        im = ax_xz.imshow(data[:, mid, :], origin="lower", extent=extent_xz, aspect="equal", cmap="magma")
        ax_xz.set_xlabel(r"$\Delta q_x$")
        ax_xz.set_ylabel(r"$\Delta q_z$")
        ax_xz.set_title("$q_y = 0$ slice")
        colorbars.append(fig.colorbar(im, ax=ax_xz, shrink=0.75))

        extent_yz = [qy[0], qy[-1], qz[0], qz[-1]]
        im = ax_yz.imshow(data[:, :, mid], origin="lower", extent=extent_yz, aspect="equal", cmap="magma")
        ax_yz.set_xlabel(r"$\Delta q_y$")
        ax_yz.set_ylabel(r"$\Delta q_z$")
        ax_yz.set_title("$q_x = 0$ slice")
        colorbars.append(fig.colorbar(im, ax=ax_yz, shrink=0.75))

        qr_3d = np.sqrt(
            qx[np.newaxis, np.newaxis, :] ** 2
            + qy[np.newaxis, :, np.newaxis] ** 2
            + qz[:, np.newaxis, np.newaxis] ** 2,
        )
        n_bins = N // 2
        bin_edges = np.linspace(0, qr_3d.max(), n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        digitised = np.digitize(qr_3d.ravel(), bin_edges) - 1
        flat = intensity.ravel()
        radial = np.zeros(n_bins)
        for i in range(n_bins):
            mask = digitised == i
            if mask.any():
                radial[i] = flat[mask].mean()
        ax_rad.semilogy(bin_centres, radial + 1, "C0", lw=1.2)
        ax_rad.set_xlabel(r"$|\Delta q|$ [$\AA^{-1}$]")
        ax_rad.set_ylabel(r"$\langle I \rangle$")
        ax_rad.set_title("Radial average")

        h, k, l = params.hkl
        fig.suptitle(
            f"BCDI forward model — ({h}{k}{l}),  "
            f"size={params.particle_size_nm:.0f} nm,  "
            f"ε_xx={params.strain_voigt[0]:.1e}",
            fontsize=11,
        )

    draw()

    # Equations / info panel
    eq_text = (
        "BCDI forward model\n"
        "──────────────────────────\n"
        "ψ(r) = ρ(r) exp(i G·ε·r)\n"
        "I(Q) = |FFT[ψ]|²\n"
        "G = h b₁ + k b₂ + l b₃\n"
        "u(r) = ε · r\n"
        f"Δr = {params.voxel_size_nm:.1f} nm\n"
        f"N  = {params.grid_size}"
    )
    fig.text(
        0.64, 0.97, eq_text,
        fontsize=9, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.6),
    )

    # Derived values
    info_text = fig.text(
        0.64, 0.62,
        f"E = {params.energy_keV:.1f} keV\n"
        f"λ = {params.wavelength_A:.4f} Å\n"
        f"d = {params.d_spacing_A:.4f} Å\n"
        f"2θ = {2 * params.theta_B_deg:.3f}°\n"
        f"|G| = {np.linalg.norm(params.G_hkl):.4f} Å⁻¹",
        fontsize=9, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.5),
    )

    # Sliders
    slider_specs = [
        ("Particle size [nm]", 50.0, 600.0, params.particle_size_nm),
        ("ε_xx  (×10⁻³)",     -5.0,   5.0, params.strain_voigt[0] * 1e3),
        ("ε_yy  (×10⁻³)",     -5.0,   5.0, params.strain_voigt[1] * 1e3),
        ("ε_zz  (×10⁻³)",     -5.0,   5.0, params.strain_voigt[2] * 1e3),
        ("log₁₀(photons)",     2.0,   8.0,  5.0),
    ]

    slider_top = 0.45
    slider_step = 0.05
    sliders: list[Slider] = []
    for i, (label, vmin, vmax, vinit) in enumerate(slider_specs):
        ax_s = fig.add_axes([0.64, slider_top - i * slider_step, 0.30, 0.025])
        s = Slider(ax_s, label, vmin, vmax, valinit=vinit)
        sliders.append(s)

    def update(_):
        size_nm = sliders[0].val
        exx = sliders[1].val * 1e-3
        eyy = sliders[2].val * 1e-3
        ezz = sliders[3].val * 1e-3
        photons = 10.0 ** sliders[4].val

        params.particle_size_nm = size_nm
        params.strain_voigt = (exx, eyy, ezz, 0.0, 0.0, 0.0)
        params.max_photons = photons

        draw()

        info_text.set_text(
            f"E = {params.energy_keV:.1f} keV\n"
            f"λ = {params.wavelength_A:.4f} Å\n"
            f"d = {params.d_spacing_A:.4f} Å\n"
            f"2θ = {2 * params.theta_B_deg:.3f}°\n"
            f"|G| = {np.linalg.norm(params.G_hkl):.4f} Å⁻¹"
        )
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    plt.show()


# ---------------------------------------------------------------------------
#  Pretty-print summary
# ---------------------------------------------------------------------------
def print_summary(params: BCDIForwardParams):
    params._recompute()
    h, k, l = params.hkl
    print("=" * 58)
    print("  BCDI Forward Model")
    print("=" * 58)
    print()
    print("  Beam")
    print(f"    Energy          : {params.energy_keV:.2f} keV")
    print(f"    Wavelength      : {params.wavelength_A:.4f} Å")
    print()
    print("  Lattice")
    print(f"    a, b, c         : {params.a:.4f}, {params.b:.4f}, {params.c:.4f} Å")
    print(f"    α, β, γ         : {params.alpha_deg:.1f}, {params.beta_deg:.1f}, {params.gamma_deg:.1f}°")
    print(f"    Reflection      : ({h} {k} {l})")
    print(f"    d-spacing       : {params.d_spacing_A:.4f} Å")
    print(f"    |G_hkl|         : {np.linalg.norm(params.G_hkl):.4f} Å⁻¹")
    print(f"    Bragg angle θ_B : {params.theta_B_deg:.3f}°")
    print(f"    2θ              : {2 * params.theta_B_deg:.3f}°")
    print()
    print("  Grid")
    print(f"    N               : {params.grid_size}")
    print(f"    Voxel size      : {params.voxel_size_nm:.1f} nm")
    print(f"    Δq              : {params.delta_q_inv_A:.6f} Å⁻¹")
    print(f"    q-range         : ±{params.grid_size * params.delta_q_inv_A / 2:.4f} Å⁻¹")
    print()
    print("  Particle")
    print(f"    Shape           : {params.particle_shape}")
    print(f"    Size            : {params.particle_size_nm:.1f} nm")
    print()
    eps = params.strain_tensor
    print("  Strain tensor")
    for row in eps:
        print(f"    [{row[0]:+.2e}  {row[1]:+.2e}  {row[2]:+.2e}]")
    print()
    if params.max_photons is not None:
        print(f"  Noise: Poisson, max_photons = {params.max_photons:.1e}")
    else:
        print("  Noise: none")
    print("=" * 58)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BCDI forward model: 3D coherent diffraction pattern via FFT.",
    )
    p.add_argument("--energy", type=float, default=12.4, help="X-ray energy [keV]")
    p.add_argument("--hkl", type=int, nargs=3, default=[1, 1, 0], metavar=("H", "K", "L"))
    p.add_argument(
        "--lattice", type=float, nargs=6, default=[6.13, 6.13, 6.13, 90, 90, 90],
        metavar=("a", "b", "c", "alpha", "beta", "gamma"),
        help="Lattice params: a b c [Å], alpha beta gamma [°]",
    )
    p.add_argument("--grid-size", type=int, default=128, help="N for N×N×N grid")
    p.add_argument("--voxel-size", type=float, default=5.0, help="Voxel size [nm]")
    p.add_argument("--shape", choices=["sphere", "cube", "faceted"], default="sphere")
    p.add_argument("--particle-size", type=float, default=300.0, help="Particle size [nm]")
    p.add_argument(
        "--strain", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0],
        metavar=("xx", "yy", "zz", "yz", "xz", "xy"),
        help="Strain tensor in Voigt notation",
    )
    p.add_argument("--photons", type=float, default=None, help="Max photons for Poisson noise")
    p.add_argument("--interactive", action="store_true", help="Launch interactive slider GUI")
    p.add_argument("--save", type=str, default=None, help="Save intensity + q-axes to .npz")
    return p.parse_args()


def main():
    args = _parse_args()
    lat = args.lattice

    params = BCDIForwardParams(
        energy_keV=args.energy,
        a=lat[0], b=lat[1], c=lat[2],
        alpha_deg=lat[3], beta_deg=lat[4], gamma_deg=lat[5],
        hkl=tuple(args.hkl),
        grid_size=args.grid_size if not args.interactive else min(args.grid_size, 64),
        voxel_size_nm=args.voxel_size,
        particle_shape=args.shape,
        particle_size_nm=args.particle_size,
        strain_voigt=tuple(args.strain),
        max_photons=args.photons,
    )

    print_summary(params)

    if args.interactive:
        interactive_plot(params)
        return

    intensity, q_axes = forward_model(params)
    print(f"\nIntensity shape: {intensity.shape}")
    print(f"Max intensity:   {intensity.max():.4e}")

    if args.save:
        qz, qy, qx = q_axes
        np.savez_compressed(
            args.save,
            intensity=intensity,
            qx=qx, qy=qy, qz=qz,
            hkl=np.array(params.hkl),
            energy_keV=params.energy_keV,
            voxel_size_nm=params.voxel_size_nm,
            strain_voigt=np.array(params.strain_voigt),
        )
        print(f"Saved to {args.save}")

    plot_forward_model(intensity, q_axes, params)
    plt.show()


if __name__ == "__main__":
    main()
