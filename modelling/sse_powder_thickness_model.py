"""
sse_powder_thin_model.py

Estimate the **maximum** powder film thickness for single-crystal Bragg XPCS.

With translation and rotation motors, finding a diffracting grain is
trivial (scan until one appears).  The binding constraint is the upper
limit: the film must be thin enough that at any given beam position and
sample angle, the number of simultaneously diffracting grains N_Bragg
stays below N_max -- otherwise spots merge into a powder ring and
speckle contrast is lost.

Physics:
    L       = t / sin(θ_B)                     (path length through film)
    N_Bragg = (φ · A_beam · L / v_grain) · (m_hkl · cos(θ_B) · Δω / 2)
    t_max   = thickness at which N_Bragg = N_max

Grain count uses an absorption-weighted effective number N_eff:
    - Grains uniformly distributed in depth 0..t → average depth t/2.
    - Symmetric Bragg reflection: incident + exit path → intensity weight
      w(z) = exp(-2µz/sin(θ_B)) per grain at depth z.
    - N_eff(t) = n_z · (2/k) · tanh(kt/2), with k = 2µ/sin(θ_B) [µm⁻¹].
    - For µ→0 this recovers the geometric count; for strong absorption
      N_eff saturates at an attenuation-length-limited value.

where Δω is the effective rocking-curve width (mosaic ⊕ Darwin ⊕ beam
divergence), and v_grain = (4/3)π(d/2)³.  The beam footprint on the
sample surface is an ellipse with short axis D and long axis D/sin(θ_B).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
#  Physical constants
# ---------------------------------------------------------------------------
HC_KEV_A = 12.3984193          # hc in keV·Å


# ---------------------------------------------------------------------------
#  Parameter dataclass
# ---------------------------------------------------------------------------
@dataclass
class BraggXPCSParams:
    """All tuneable parameters for the powder-thickness model."""

    # --- Beam ---
    energy_keV: float = 12.4                # X-ray energy [keV]
    beam_diameter_um: float = 1.0           # beam diameter [µm] (circular profile)

    # --- Sample / powder ---
    grain_diameter_um: float = 1.0          # crystallite diameter [µm]
    packing_fraction: float = 0.50          # volume fraction of grains
    mosaic_spread_deg: float = 0.05         # grain mosaic spread [°]

    # --- Crystal structure / reflection ---
    d_spacing_A: float = 3.0                # d-spacing of (hkl) [Å]
    multiplicity: int = 4                   # m_hkl  (4 for (110) in ortho/mono)

    # --- Beamline optics ---
    beam_divergence_urad: float = 32.0      # beam divergence [µrad] (1µm pinhole, 12.4keV)
    darwin_width_urad: float = 5.0          # intrinsic Darwin width [µrad] (perfect crystal)

    # --- Absorption (optional) ---
    mu_cm_inv: float = 0.023                # linear absorption coeff [cm⁻¹] (Na₂B₁₀H₁₀ @ 12.4keV)

    # --- XPCS criterion ---
    N_max: float = 10.0                     # max acceptable diffracting grains
    detector_distance_m: float = 2.0       # sample-to-detector distance [m]

    # --- Derived (computed in __post_init__) ---
    wavelength_A: float = field(init=False)
    beam_area_um2: float = field(init=False)
    grain_volume_um3: float = field(init=False)
    theta_B_rad: float = field(init=False)
    theta_B_deg: float = field(init=False)
    delta_omega_rad: float = field(init=False)
    footprint_long_um: float = field(init=False)  # ellipse long axis on surface
    spot_size_deg: float = field(init=False)      # azimuthal extent of one Bragg spot [°]

    def __post_init__(self):
        self._recompute()

    def _recompute(self):
        """Recompute all derived quantities from the primary parameters."""
        self.wavelength_A = HC_KEV_A / self.energy_keV
        self.beam_area_um2 = np.pi * (self.beam_diameter_um / 2.0) ** 2
        r = self.grain_diameter_um / 2.0
        self.grain_volume_um3 = (4.0 / 3.0) * np.pi * r**3

        # Bragg angle from Bragg's law: λ = 2 d sin(θ)
        sin_theta = self.wavelength_A / (2.0 * self.d_spacing_A)
        if abs(sin_theta) > 1.0:
            raise ValueError(
                f"Bragg condition impossible: λ={self.wavelength_A:.4f} Å, "
                f"d={self.d_spacing_A:.4f} Å  →  sin(θ)={sin_theta:.4f}"
            )
        self.theta_B_rad = np.arcsin(sin_theta)
        self.theta_B_deg = np.degrees(self.theta_B_rad)

        # Effective rocking-curve width (quadrature sum)
        eta_rad = np.radians(self.mosaic_spread_deg)
        div_rad = self.beam_divergence_urad * 1e-6
        dar_rad = self.darwin_width_urad * 1e-6
        self.delta_omega_rad = np.sqrt(eta_rad**2 + div_rad**2 + dar_rad**2)

        # Beam footprint on sample surface (ellipse long axis)
        self.footprint_long_um = self.beam_diameter_um / np.sin(self.theta_B_rad)

        # Azimuthal spot size on Debye-Scherrer ring
        # Dominated by angular spread (mosaic + divergence) projected azimuthally
        # For small angles, azimuthal spread ≈ angular spread
        # Add detector resolution contribution (~50µm pixel at 2m → ~0.02° per pixel,
        # typical spot ~3-5 pixels → ~0.1° contribution)
        pixel_size_um = 50.0  # typical XPCS detector pixel size
        detector_res_deg = np.degrees(pixel_size_um * 1e-6 / (self.detector_distance_m * np.tan(2 * self.theta_B_rad))) * 5  # ~5 pixels per spot
        angular_spread_deg = np.degrees(self.delta_omega_rad)
        self.spot_size_deg = np.sqrt(angular_spread_deg**2 + detector_res_deg**2)


# ---------------------------------------------------------------------------
#  Core computation
# ---------------------------------------------------------------------------
def n_bragg_vs_thickness(params: BraggXPCSParams,
                         t_um: np.ndarray) -> np.ndarray:
    """
    Expected number of diffracting grains as a function of film thickness.

    The beam path through the film is L = t / sin(θ_B), accounting for
    the elongated footprint at grazing incidence.

    Parameters
    ----------
    params : BraggXPCSParams
    t_um   : array of thicknesses [µm]

    Returns
    -------
    N_Bragg : array, same shape as t_um
    """
    path_um = t_um / np.sin(params.theta_B_rad)
    N_total = (params.packing_fraction * params.beam_area_um2 * path_um
               / params.grain_volume_um3)
    P_bragg = (params.multiplicity * np.cos(params.theta_B_rad)
               * params.delta_omega_rad / 2.0)
    return N_total * P_bragg


def _absorption_decay_constant_per_um(params: BraggXPCSParams) -> float:
    """
    Decay constant k [µm⁻¹] for depth weighting w(z) = exp(-k·z).

    Symmetric Bragg: incident + exit path gives 2µ·(z/sin(θ_B)) in cm,
    so k_um = 2·µ_cm⁻¹·(1e-4)/sin(θ_B).
    """
    return 2.0 * params.mu_cm_inv * 1e-4 / np.sin(params.theta_B_rad)


def _grains_per_um_depth(params: BraggXPCSParams) -> float:
    """Grains per µm depth (along surface normal): n_z = (φ A/v)·(1/sin θ)·P_Bragg."""
    P_bragg = (params.multiplicity * np.cos(params.theta_B_rad)
               * params.delta_omega_rad / 2.0)
    return (params.packing_fraction * params.beam_area_um2
            / params.grain_volume_um3) * (1.0 / np.sin(params.theta_B_rad)) * P_bragg


def n_bragg_effective_vs_thickness(params: BraggXPCSParams,
                                    t_um: np.ndarray) -> np.ndarray:
    """
    Absorption-weighted effective number of diffracting grains N_eff(t).

    Assumes grains uniformly distributed in depth 0..t and symmetric
    Bragg reflection (incident + exit path). Intensity weight per grain
    at depth z: w(z) = exp(-k·z) with k = 2µ/sin(θ_B) [µm⁻¹]. Then

        N_eff(t) = n_z · (2/k) · tanh(k·t/2).

    For µ→0 this reduces to the geometric count n_z·t; for strong
    absorption N_eff saturates. Used for speckle contrast and t_max.

    Parameters
    ----------
    params : BraggXPCSParams
    t_um   : array of thicknesses [µm]

    Returns
    -------
    N_eff : array, same shape as t_um
    """
    k_um = _absorption_decay_constant_per_um(params)
    n_z = _grains_per_um_depth(params)
    # N_eff = n_z * (2/k) * tanh(k*t/2); avoid div by zero when µ=0
    if k_um <= 0:
        return n_bragg_vs_thickness(params, t_um)
    factor = 2.0 / k_um
    return n_z * factor * np.tanh(k_um * t_um / 2.0)


def absorption_transmission(params: BraggXPCSParams,
                            t_um: np.ndarray) -> np.ndarray:
    """
    Two-pass transmission: intensity after incident + reflected path through
    full thickness. T = exp(-2µ L), with L = t/sin(θ_B) in cm.

    Parameters
    ----------
    params : BraggXPCSParams
    t_um   : array of thicknesses [µm]

    Returns
    -------
    T : array, same shape as t_um
    """
    path_cm = (t_um / np.sin(params.theta_B_rad)) * 1e-4
    return np.exp(-2.0 * params.mu_cm_inv * path_cm)


def speckle_contrast(params: BraggXPCSParams,
                     t_um: np.ndarray) -> np.ndarray:
    """
    Speckle contrast  β = 1 / max(N_eff, 1).

    Uses absorption-weighted N_eff (depth-averaged, two-pass attenuation).
    For a single effectively contributing grain β = 1; contrast drops as 1/N.
    """
    N = n_bragg_effective_vs_thickness(params, t_um)
    return 1.0 / np.maximum(N, 1.0)


def overlap_probability(params: BraggXPCSParams,
                        t_um: np.ndarray) -> np.ndarray:
    """
    Birthday-problem probability that at least two Bragg spots overlap
    on the Debye-Scherrer ring. Uses N_eff.

    M = 360 / spot_size_deg  independent azimuthal slots.
    P(overlap) ≈ 1 - exp(-N(N-1) / (2M))
    """
    M = 360.0 / params.spot_size_deg
    N = n_bragg_effective_vs_thickness(params, t_um)
    return 1.0 - np.exp(-N * (N - 1.0) / (2.0 * M))


def max_thickness_Nmax(params: BraggXPCSParams) -> float:
    """
    Maximum film thickness [µm] such that N_eff ≤ N_max.

    Solves N_eff(t) = n_z·(2/k)·tanh(kt/2) = N_max for t.
    With motors, finding grains is trivial; constraint is the upper limit.
    """
    params._recompute()
    k_um = _absorption_decay_constant_per_um(params)
    n_z = _grains_per_um_depth(params)
    if k_um <= 0:
        sin_theta = np.sin(params.theta_B_rad)
        P_bragg = (params.multiplicity * np.cos(params.theta_B_rad)
                   * params.delta_omega_rad / 2.0)
        grains_per_um_path = (params.packing_fraction * params.beam_area_um2
                              / params.grain_volume_um3)
        return params.N_max * sin_theta / (grains_per_um_path * P_bragg)
    rhs = params.N_max * k_um / (2.0 * n_z)
    if rhs >= 1.0:
        return np.inf
    rhs = min(rhs, 1.0 - 1e-10)
    return (2.0 / k_um) * np.arctanh(rhs)


def max_thickness_contrast(params: BraggXPCSParams) -> float:
    """
    Film thickness [µm] where N_eff = 1 (β = 1, perfect contrast).

    Solves N_eff(t) = 1 for t. Above this thickness, contrast drops.
    """
    params._recompute()
    k_um = _absorption_decay_constant_per_um(params)
    n_z = _grains_per_um_depth(params)
    if k_um <= 0:
        sin_theta = np.sin(params.theta_B_rad)
        P_bragg = (params.multiplicity * np.cos(params.theta_B_rad)
                   * params.delta_omega_rad / 2.0)
        grains_per_um_path = (params.packing_fraction * params.beam_area_um2
                              / params.grain_volume_um3)
        return sin_theta / (grains_per_um_path * P_bragg)
    rhs = 1.0 * k_um / (2.0 * n_z)
    if rhs >= 1.0:
        return np.inf
    rhs = min(rhs, 1.0 - 1e-10)
    return (2.0 / k_um) * np.arctanh(rhs)


# ---------------------------------------------------------------------------
#  Pretty-print summary
# ---------------------------------------------------------------------------
def print_summary(params: BraggXPCSParams):
    params._recompute()
    t_max_Nmax = max_thickness_Nmax(params)
    t_max_contrast = max_thickness_contrast(params)

    print("=" * 62)
    print("  SSE Powder Film Thickness Model for Bragg XPCS")
    print("=" * 62)
    print()
    print("  Beam parameters")
    print(f"    Energy              : {params.energy_keV:.2f} keV")
    print(f"    Wavelength          : {params.wavelength_A:.4f} Å")
    print(f"    Beam diameter       : {params.beam_diameter_um:.1f} µm")
    print(f"    Beam area           : {params.beam_area_um2:.2f} µm²")
    print(f"    Footprint (long)    : {params.footprint_long_um:.2f} µm")
    print(f"    Beam divergence     : {params.beam_divergence_urad:.1f} µrad")
    print()
    print("  Crystal / reflection")
    print(f"    d-spacing           : {params.d_spacing_A:.4f} Å")
    print(f"    Bragg angle θ_B     : {params.theta_B_deg:.3f}°")
    print(f"    2θ                  : {2 * params.theta_B_deg:.3f}°")
    print(f"    Multiplicity m_hkl  : {params.multiplicity}")
    print(f"    Darwin width        : {params.darwin_width_urad:.1f} µrad")
    print()
    print("  Sample")
    print(f"    Grain diameter      : {params.grain_diameter_um:.2f} µm")
    print(f"    Grain volume        : {params.grain_volume_um3:.2f} µm³")
    print(f"    Packing fraction    : {params.packing_fraction:.2f}")
    print(f"    Mosaic spread       : {params.mosaic_spread_deg:.4f}°")
    print(f"    Absorption coeff µ  : {params.mu_cm_inv:.4f} cm⁻¹")
    print()
    print("  Effective rocking width")
    print(f"    Δω (total)          : {np.degrees(params.delta_omega_rad):.5f}° "
          f"({params.delta_omega_rad * 1e6:.1f} µrad)")
    print()
    print("  >>> MAXIMUM THICKNESS (N_Bragg = {:.0f})".format(params.N_max))
    print(f"      t_max_Nmax = {t_max_Nmax:.1f} µm  ({t_max_Nmax * 1e-3:.3f} mm)")
    print(f"      t_max_contrast = {t_max_contrast:.1f} µm  ({t_max_contrast * 1e-3:.3f} mm)")
    print()
    print("  (With motors, there is no minimum thickness constraint.)")
    print("=" * 62)


# ---------------------------------------------------------------------------
#  Interactive plot
# ---------------------------------------------------------------------------
def interactive_plot(params: BraggXPCSParams | None = None):
    """
    Launch an interactive matplotlib figure with sliders for all key
    parameters.  Three panels:

        1. N_Bragg(t)   with shaded feasible window
        2. Speckle contrast β(t) = 1/N_Bragg
        3. Debye-Scherrer ring with spots (polar plot)
    """
    if params is None:
        params = BraggXPCSParams()

    fig = plt.figure(figsize=(11.5, 8))
    fig.subplots_adjust(left=0.10, right=0.55, bottom=0.08, top=0.95,
                        hspace=0.35)

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3, polar=True)
    # Move ring plot down slightly to make room for title below
    pos = ax3.get_position()
    ax3.set_position([pos.x0, pos.y0 - 0.02, pos.width, pos.height])

    # Random generator for spot positions (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    # Spot shape on polar plot: [phi half-width (°), r inner, r outer]. Set here; no sliders.
    spot_shape = [3.0, 0.94, 1.06]

    def draw_panels():
        """Clear and redraw all three plot panels from current params."""
        params._recompute()
        t_max_Nmax = max_thickness_Nmax(params)
        t_max_contrast = max_thickness_contrast(params)
        # Plot range: small t to 10% beyond t_max_Nmax (no fixed 100 µm floor)
        t_max_plot = t_max_Nmax * 1.1
        t = np.linspace(0.1, t_max_plot, 2000)
        
        N_eff = n_bragg_effective_vs_thickness(params, t)
        N_geom = n_bragg_vs_thickness(params, t)
        beta = speckle_contrast(params, t)

        # Panel 1: N_eff vs thickness (with geometric N as reference)
        ax1.cla()
        ax1.semilogy(t, N_eff, "C0", lw=2, label="$N_{\\rm eff}$ (absorption-weighted)")
        ax1.semilogy(t, N_geom, "C0", lw=0.8, ls="--", alpha=0.6, label="$N_{\\rm Bragg}$ (geometric)")
        ax1.axhline(params.N_max, color="C2", ls="--", lw=0.8,
                     label=f"$N_{{max}}$ = {params.N_max:.0f}")
        if t[0] <= t_max_Nmax <= t[-1]:
            ax1.axvline(t_max_Nmax, color="C2", lw=1.5, ls="-",
                         label=f"$t_{{max,Nmax}}$ = {t_max_Nmax:.1f} µm")
        ax1.axvspan(0, min(t_max_Nmax, t[-1]), alpha=0.08, color="C2")
        ax1.set_ylabel("$N_{\\rm eff}$, $N_{\\rm Bragg}$")
        ax1.set_title("Effective number of diffracting grains (N_eff: depth + two-pass absorption)")
        ax1.legend(loc="lower center", fontsize=8)
        ax1.set_ylim(1e-2, None)

        # Panel 2: Speckle contrast vs thickness
        ax2.cla()
        ax2.plot(t, beta, "C1", lw=2)
        if t[0] <= t_max_contrast <= t[-1]:
            ax2.axvline(t_max_contrast, color="C4", lw=1.5, ls="-",
                         label=f"$t_{{max,contrast}}$ = {t_max_contrast:.1f} µm")
        ax2.axvspan(0, min(t_max_contrast, t[-1]), alpha=0.08, color="C4")
        ax2.set_ylabel("Speckle contrast  $\\beta$")
        ax2.set_xlabel("Film thickness $t$  [µm]")
        ax2.set_title("Speckle contrast  $\\beta = 1 / N_{\\rm eff}$")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc="upper right", fontsize=8)

        # Panel 3: Spots on the Debye-Scherrer ring at t_max (shape from spot_shape)
        spot_streak_phi, spot_radial_inner, spot_radial_outer = spot_shape[0], spot_shape[1], spot_shape[2]
        ax3.cla()
        N_at_tmax = max(1, int(round(params.N_max)))
        # spot_streak_phi = azimuthal half-width in degrees (so 10 → 20° arc); directly sets drawn size
        phi_half = np.radians(spot_streak_phi)
        spot_centers = rng.uniform(0, 2 * np.pi, size=N_at_tmax)
        arc_t = np.linspace(-phi_half, phi_half, 40)
        ax3.plot(np.linspace(0, 2 * np.pi, 200),
                 np.ones(200), color="lightgray", lw=1)
        for phi_c in spot_centers:
            ax3.fill_between(phi_c + arc_t, spot_radial_inner, spot_radial_outer,
                             alpha=0.6, color="C0")
        ax3.set_ylim(0, 1.6)
        ax3.set_yticks([])
        
        return t_max_Nmax, t_max_contrast, N_at_tmax

    # Initial draw
    t_max_Nmax, t_max_contrast, N_at_tmax = draw_panels()
    
    # Title for panel 3 (below the polar plot)
    ax3_pos = ax3.get_position()
    ring_title = fig.text(
        ax3_pos.x0 + ax3_pos.width / 2, ax3_pos.y0 - 0.035,
        f"Ring at $t_{{max,Nmax}}$  "
        f"({N_at_tmax} spot{'s' if N_at_tmax != 1 else ''}, "
        f"Δφ={params.spot_size_deg:.1f}°)",
        ha='center', va='top', fontsize=9,
    )

    # Governing equations (static)
    eq_lines = (
        "Governing equations\n"
        "───────────────────────────────────\n"
        "λ = hc / E\n"
        "sin(θ_B) = λ / (2d)\n"
        "Δω = √(η² + δ_div² + δ_Dar²)\n"
        "A = π(D_beam/2)²\n"
        "L = t / sin(θ_B)        [path length]\n"
        "k = 2µ/sin(θ_B) [µm⁻¹] [two-pass]\n"
        "n_z = (φ·A/v)·(1/sin θ)·P_Bragg\n"
        "N_eff = n_z·(2/k)·tanh(kt/2)\n"
        "β = 1 / max(N_eff, 1)\n"
        "Δφ_spot ≈ √(Δω² + (p·n/L·tan(2θ))²)\n"
        "t_max : N_eff = N_max or 1"
    )
    fig.text(
        0.655, 0.97, eq_lines,
        fontsize=9, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.6),
    )

    # Dynamic derived-values annotation (below equations)
    bragg_text = fig.text(
        0.65, 0.67,
        f"θ_B = {params.theta_B_deg:.3f}°    "
        f"2θ = {2 * params.theta_B_deg:.3f}°\n"
        f"λ  = {params.wavelength_A:.4f} Å    "
        f"Δω = {np.degrees(params.delta_omega_rad):.5f}°\n"
        f"Footprint = {params.footprint_long_um:.2f} µm\n"
        f"Δφ_spot = {params.spot_size_deg:.2f}°\n\n"
        f">>> t_max_Nmax = {t_max_Nmax:.1f} µm ({t_max_Nmax*1e-3:.3f} mm)\n"
        f">>> t_max_contrast = {t_max_contrast:.1f} µm ({t_max_contrast*1e-3:.3f} mm)",
        fontsize=9, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.5),
    )

    # ---- Sliders ----
    slider_specs = [
        ("Energy [keV]",        1.0,   30.0,  params.energy_keV),
        ("Grain diam [µm]",     0.1,   10.0,  params.grain_diameter_um),
        ("Beam diam [µm]",      0.1,   10.0,  params.beam_diameter_um),
        ("Packing frac",        0.05,  0.74, params.packing_fraction),
        ("Mosaic [°]",          0.001, 0.3,  params.mosaic_spread_deg),
        ("d-spacing [Å]",       0.5,   20.0,  params.d_spacing_A),
        ("Multiplicity",        1,     48,    params.multiplicity),
        ("Divergence [µrad]",   1.0,   50.0,  params.beam_divergence_urad),
        ("Darwin [µrad]",       1.0,   10.0,  params.darwin_width_urad),
        ("µ [cm⁻¹]",            0.001,  0.05,  params.mu_cm_inv),
        ("N_max",               1.0,   50.0,  params.N_max),
        ("Detector dist [m]",   0.1,    4.0,  params.detector_distance_m),
    ]

    slider_top = 0.48
    slider_step = 0.04
    sliders = []
    for i, (label, vmin, vmax, vinit) in enumerate(slider_specs):
        ax_s = fig.add_axes([0.69, slider_top - i * slider_step, 0.25, 0.025])
        is_int = label == "Multiplicity"
        s = Slider(ax_s, label, vmin, vmax, valinit=vinit,
                   valstep=1 if is_int else None)
        sliders.append(s)

    def update(val):
        vals = [s.val for s in sliders]
        (params.energy_keV, params.grain_diameter_um,
         params.beam_diameter_um,
         params.packing_fraction, params.mosaic_spread_deg,
         params.d_spacing_A, params.multiplicity,
         params.beam_divergence_urad, params.darwin_width_urad,
         params.mu_cm_inv, params.N_max,
         params.detector_distance_m) = vals[:12]
        params.multiplicity = int(params.multiplicity)

        try:
            t_max_Nmax, t_max_contrast, N_at_tmax = draw_panels()
        except ValueError:
            return

        bragg_text.set_text(
            f"θ_B = {params.theta_B_deg:.3f}°    "
            f"2θ = {2 * params.theta_B_deg:.3f}°\n"
            f"λ  = {params.wavelength_A:.4f} Å    "
            f"Δω = {np.degrees(params.delta_omega_rad):.5f}°\n"
            f"Footprint = {params.footprint_long_um:.2f} µm\n"
            f"Δφ_spot = {params.spot_size_deg:.2f}°\n\n"
            f">>> t_max_Nmax = {t_max_Nmax:.1f} µm ({t_max_Nmax*1e-3:.3f} mm)\n"
            f">>> t_max_contrast = {t_max_contrast:.1f} µm ({t_max_contrast*1e-3:.3f} mm)"
        )
        
        ring_title.set_text(
            f"Ring at $t_{{max,Nmax}}$  "
            f"({N_at_tmax} spot{'s' if N_at_tmax != 1 else ''}, "
            f"Δφ={params.spot_size_deg:.1f}°)"
        )

        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    plt.show()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    params = BraggXPCSParams(
        energy_keV=12.4,
        beam_diameter_um=1,
        grain_diameter_um=1.0,
        packing_fraction=0.50,
        mosaic_spread_deg=0.05,
        d_spacing_A=5.73,
        multiplicity=4,
        beam_divergence_urad=25.0,
        darwin_width_urad=5.0,
        mu_cm_inv=0.023,
        N_max=2.0,
        detector_distance_m=2.0,
    )

    print_summary(params)
    interactive_plot(params)
