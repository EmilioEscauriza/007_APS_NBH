"""
TTC (two-time correlation) plots from the transport-coefficient analytic model.

Implements the XPCS two-time intensity autocorrelation c2(q, t1, t2) from:
  - He et al. 2024, PNAS: "Transport coefficient approach for characterizing
    nonequilibrium dynamics in soft matter"
  - He et al. 2025, PNAS: "Bridging microscopic dynamics and rheology in the
    yielding of charged colloidal suspensions"

Two flow regimes:
  1. Laminar (homodyne): homogeneous shear, Eq. 13 (2024) / Eq. 1 (2025)
  2. Heterodyne: two components (static + flowing), Eq. 14 (2024) / Eq. 2–3 (2025)

Time axis is set by DEFAULT_TIME_RANGE and DEFAULT_DT_S (same convention as fit script).
Usage:
  python ttc_plots_transport_coefficient.py              # defaults from script
  python ttc_plots_transport_coefficient.py --laminar
  python ttc_plots_transport_coefficient.py --heterodyne
  python ttc_plots_transport_coefficient.py --time-range 1000,3000,2 --dt-s 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
#  Data (for side-by-side real vs model); same as ttc_fit_transport_coefficient.py
# ---------------------------------------------------------------------------
SAMPLE_ID = "A073"
MASK_N = 145
POSITION_NAME = "A4"


# ---------------------------------------------------------------------------
#  Default CLI options (edit these; overridden by command-line args)
#  Time axis matches fit script: TIME_RANGE = "full" or [index_start, index_end, stride];
#  DT_S = seconds per step (frame).
# ---------------------------------------------------------------------------
DEFAULT_LAMINAR = False
DEFAULT_HETERODYNE = True
DEFAULT_SAVE = False
DEFAULT_NO_SHOW = False
# e.g. [1000, 3000, 2] → indices 1000:3000:2 → 1000 points; or "full" → default 1000 points
DEFAULT_TIME_RANGE = [0, 4800, 1]
DEFAULT_DT_S = 0.5
# Heterodyne stripe frequency: period (s) of one oscillation along delay. Smaller = more lines.
# v = 2π/(q*period); e.g. 175 s → ~0.67 Å/s, 100 s → ~1.2 Å/s.
DEFAULT_STRIPE_PERIOD_S = 350.0
# If True, J(t) and x_s(t) are constant so there is no linear ramp (dim TL → bright BR) perpendicular
# to t1=t2. v(t) still has two regimes so stripe frequency changes with time (non-flat).
# If False, J(t) and x_s(t) vary with t → you get that ramp on top of the stripes.
DEFAULT_CONSTANT_J_AND_XS = False
# Periodic v(t): if True, v(t) = v_mean * (1 + amp*cos(2π t/T)) instead of tanh two-regime.
DEFAULT_V_PERIODIC = True
DEFAULT_V_PERIOD_S = 300.0
DEFAULT_V_AMP_FRAC = 0.2  # amplitude as fraction of v_mean; keep < 1 so v stays positive
# Time-varying stripe period (when periodic v): T(t) = A + B*cos(π t/L + φ), L=2400 s; T(0)=205, T(1200)=415, T(2400)=255 s.
STRIPE_PERIOD_COSINE_L_S = 2400.0
STRIPE_PERIOD_COSINE_A = 230.0
STRIPE_PERIOD_COSINE_B = np.sqrt(25**2 + 185**2)
STRIPE_PERIOD_COSINE_PHI = np.arctan2(-185.0, -25.0)
# Time-dependent J(t) and x_s(t): time constants as fraction of t_max (forward-time saturation).
DEFAULT_XS_TAU_FRAC = 1  # x_s(t): tau = 0.25*t_max
DEFAULT_J_TAU_FRAC = 1  # J(t): tau = 0.6*t_max
# Antidiagonal decay: exp(-q²∫J dt). Smaller J = gentler decay.
DEFAULT_J_CONSTANT = 0.1   # J when constant_J_and_xs (Å²/s or same units)
DEFAULT_J0 = 1.5           # J0 when J(t) time-dependent (saturation level)


def time_axis_from_range(time_range: str | list, dt_s: float) -> np.ndarray:
    """
    Return 1D time array t [s] from TIME_RANGE and DT_S.
    time_range: "full" → 1000 points, t = 0, dt_s, 2*dt_s, ...; or
    [start, end, stride] → t[i] = (start + i*stride)*dt_s so axes show time in seconds
    (e.g. 500–1500 s for [1000, 3000, 2], dt_s=0.5).
    """
    if time_range == "full":
        n = 1000
        return np.arange(n, dtype=float) * dt_s
    start, end, stride = time_range[0], time_range[1], time_range[2]
    n = len(range(start, end, stride))
    indices = np.arange(n, dtype=float) * stride + start
    return indices * dt_s


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ttc_path(position: str, sample_id: str, mask_n: int) -> Path:
    return (
        _repo_root()
        / "data"
        / "ttc_arrays"
        / position
        / sample_id
        / f"{position}_{sample_id}_mask{mask_n:03d}_ttc_array.npy"
    )


def load_ttc(path: Path, *, symmetrize: bool = True) -> np.ndarray:
    C = np.load(path).astype(np.float64)
    if symmetrize:
        C = C + C.T - np.diag(np.diag(C))
    return C


def apply_time_range(C: np.ndarray, time_range: str | list) -> np.ndarray:
    if time_range == "full":
        return C
    start, end, stride = time_range[0], time_range[1], time_range[2]
    return C[start:end:stride, start:end:stride].copy()


# ---------------------------------------------------------------------------
#  Cumulative integral on a grid (trapezoidal)
#  Returns F so that F[j] - F[i] = integral from t[i] to t[j]
# ---------------------------------------------------------------------------
def _cumulative_integral(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """F[k] = integral from t[0] to t[k] of y(s) ds (trapezoidal)."""
    F = np.zeros_like(t)
    if len(t) < 2:
        return F
    F[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return F


def _integral_between(F: np.ndarray, t: np.ndarray, t1_2d: np.ndarray, t2_2d: np.ndarray) -> np.ndarray:
    """
    For each (i,j), return F(t2) - F(t1) by linear interpolation of F at t.
    F is defined at t[0], t[1], ...; t1_2d and t2_2d are 2D arrays (e.g. from meshgrid).
    """
    # Index into F by interpolating: index = (t - t[0]) / dt
    t_flat = t
    dt = (t_flat[-1] - t_flat[0]) / (len(t_flat) - 1) if len(t_flat) > 1 else 1.0
    if len(t_flat) == 1:
        return np.zeros_like(t1_2d)
    idx1 = (t1_2d - t_flat[0]) / dt
    idx2 = (t2_2d - t_flat[0]) / dt
    idx1 = np.clip(idx1, 0, len(F) - 1)
    idx2 = np.clip(idx2, 0, len(F) - 1)
    # Linear interpolation: F at index k is F[floor(k)] + (k - floor(k)) * (F[ceil(k)] - F[floor(k)])
    i1_lo = np.floor(idx1).astype(int)
    i1_hi = np.minimum(i1_lo + 1, len(F) - 1)
    i2_lo = np.floor(idx2).astype(int)
    i2_hi = np.minimum(i2_lo + 1, len(F) - 1)
    f1 = idx1 - i1_lo
    f2 = idx2 - i2_lo
    F1 = (1 - f1) * F[i1_lo] + f1 * F[i1_hi]
    F2 = (1 - f2) * F[i2_lo] + f2 * F[i2_hi]
    return F2 - F1


def integral_between_grid(
    y: np.ndarray, t: np.ndarray, t1_2d: np.ndarray, t2_2d: np.ndarray
) -> np.ndarray:
    """
    For each (i,j), return integral from t1_2d[i,j] to t2_2d[i,j] of y(s) ds.
    y and t are 1D (same length). Uses precomputed cumulative F = cumulative_integral(y, t).
    """
    F = _cumulative_integral(y, t)
    return _integral_between(F, t, t1_2d, t2_2d)


# ---------------------------------------------------------------------------
#  Laminar (homodyne) c2 — Eq. 13 (2024) / Eq. 1 (2025)
#  c2 = 1 + β * exp(-q² ∫ J dt) * sinc²( (1/2) q h ∫ γ̇ cos φ dt )
#  sinc(x) = sin(x)/x  →  use np.sinc(u/π) = sin(u)/u
# ---------------------------------------------------------------------------
def c2_laminar(
    t: np.ndarray,
    J: np.ndarray,
    gammadot: np.ndarray,
    *,
    q: float = 0.054,
    h: float = 1.0,
    phi_deg: float = 0.0,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Two-time correlation for laminar (homogeneous shear) flow.

    Parameters
    ----------
    t : (N,) time points [s]
    J : (N,) transport coefficient [Å²/s or same units as 1/time]
    gammadot : (N,) shear rate [1/s]
    q : scattering vector magnitude [Å⁻¹]
    h : gap between stator and rotor [mm or same length unit]
    phi_deg : angle between q and flow direction [degrees]
    beta : coherent contrast 0 ≤ β ≤ 1

    Returns
    -------
    c2 : (N, N) two-time correlation
    """
    phi = np.deg2rad(phi_deg)
    t1, t2 = np.meshgrid(t, t, indexing="xy")

    # ∫_{min(t1,t2)}^{max(t1,t2)} J(s) ds (delay-based decay: symmetric, so diagonal = max)
    int_J = np.abs(integral_between_grid(J, t, t1, t2))
    # ∫_{t1}^{t2} γ̇(s) cos φ ds
    int_g = integral_between_grid(gammadot * np.cos(phi), t, t1, t2)

    arg_sinc = 0.5 * q * h * int_g
    # sinc(x) = sin(x)/x; numpy's sinc is sin(πx)/(πx), so sin(u)/u = np.sinc(u / np.pi)
    s = np.where(np.abs(arg_sinc) < 1e-12, 1.0, np.sinc(arg_sinc / np.pi))

    c2 = 1.0 + beta * np.exp(-(q**2) * int_J) * (s**2)
    return c2


# ---------------------------------------------------------------------------
#  Heterodyne (two components: static + flowing) — Eq. 14 (2024) / Eq. 2–3 (2025)
#  x_r = fraction static, x_s = fraction flowing; x_r + x_s = 1
#  c2 = 1 + β * exp(-q² ∫ J dt) / f² * [ (x_r1 x_r2)² + (x_s1 x_s2)²
#        + 2 x_r1 x_r2 x_s1 x_s2 * cos( q cos(φ) ∫ v dt ) ]
#  f² = (x_s1² + x_r1²)(x_s2² + x_r2²)
# ---------------------------------------------------------------------------
def c2_heterodyne(
    t: np.ndarray,
    J: np.ndarray,
    v: np.ndarray,
    x_s: np.ndarray,
    *,
    q: float = 0.054,
    phi_deg: float = 0.0,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Two-time correlation for two-component (e.g. shear-banding) heterodyne.

    Parameters
    ----------
    t : (N,) time points [s]
    J : (N,) transport coefficient [Å²/s]
    v : (N,) mean velocity of flowing component [Å/s or same as q⁻¹/time]
    x_s : (N,) fraction of flowing component (static fraction x_r = 1 - x_s)
    q : scattering vector magnitude [Å⁻¹]
    phi_deg : angle between q and flow [degrees]
    beta : coherent contrast

    Returns
    -------
    c2 : (N, N) two-time correlation
    """
    phi = np.deg2rad(phi_deg)
    x_r = 1.0 - x_s

    t1, t2 = np.meshgrid(t, t, indexing="xy")

    # Delay-based decay: use |∫ J dt| so diagonal = max, decay with delay
    int_J = np.abs(integral_between_grid(J, t, t1, t2))
    int_v = integral_between_grid(v, t, t1, t2)

    # Interpolate x_r, x_s at (t1, t2) — use same grid indices as in _integral_between
    # For a regular t grid, t1[i,j] = t[i], t2[i,j] = t[j], so x_r(t1) = x_r[i], x_r(t2) = x_r[j]
    # So we need 2D arrays: x_r1[i,j] = x_r at t1 = t[i], x_r2[i,j] = x_r at t2 = t[j]
    x_r1 = np.tile(x_r[:, np.newaxis], (1, len(t)))   # (N,N): row i is x_r[i]
    x_r2 = np.tile(x_r[np.newaxis, :], (len(t), 1))   # (N,N): col j is x_r[j]
    x_s1 = np.tile(x_s[:, np.newaxis], (1, len(t)))
    x_s2 = np.tile(x_s[np.newaxis, :], (len(t), 1))

    f2 = (x_s1**2 + x_r1**2) * (x_s2**2 + x_r2**2)
    f2 = np.maximum(f2, 1e-20)

    bracket = (
        (x_r1 * x_r2) ** 2
        + (x_s1 * x_s2) ** 2
        + 2.0 * x_r1 * x_r2 * x_s1 * x_s2 * np.cos(q * np.cos(phi) * int_v)
    )

    c2 = 1.0 + beta * np.exp(-(q**2) * int_J) / f2 * bracket
    return c2


# ---------------------------------------------------------------------------
#  Example time-dependent parameter functions (for demos)
# ---------------------------------------------------------------------------
def example_laminar_params(t: np.ndarray, tmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Andrade-like creep: J decays, γ̇ decays (e.g. ~ t^{-2/3})."""
    J0 = 2.0
    g0 = 0.01
    J = J0 * (1.0 + t / tmax) ** (-1.2)
    gammadot = g0 * (1.0 + t / tmax) ** (-2.0 / 3.0)
    return J, gammadot


def example_heterodyne_params(
    t: np.ndarray,
    tmax: float,
    *,
    stripe_period_s: float = 175.0,
    q: float = 0.054,
    constant_J_and_xs: bool = True,
    use_periodic_v: bool = False,
    v_period_s: float = 100.0,
    v_amp_frac: float = 0.3,
    xs_tau_frac: float = 0.25,
    j_tau_frac: float = 0.6,
    J_constant: float = 0.5,
    J0: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shear banding: v set from stripe_period_s or time-varying T(t). v(t) is either:
    - Tanh two-regime (default): v_early → v_late so stripe frequency changes with time.
    - Periodic: T(t) = A + B*cos(π t/L + φ) over L=2400 s (205→415→255 s), then v(t) = 2π/(q*T(t)),
      with extra amplitude modulation v *= (1 + v_amp_frac*cos(2π t/v_period_s)).
    If constant_J_and_xs=True: J and x_s are constant. If False: J(t) and x_s(t) vary with t;
    time constants are tau_xs = xs_tau_frac*t_max and tau_J = j_tau_frac*t_max (forward-time saturation).
    Antidiagonal decay is exp(-q²∫J dt); lower J_constant or J0 = gentler decay.
    """
    v_mean = 2.0 * np.pi / (q * stripe_period_s)
    if use_periodic_v:
        # Time-varying stripe period T(t) = A + B*cos(π t/L + φ) over L=2400 s (205→415→255 s).
        L = STRIPE_PERIOD_COSINE_L_S
        T_t = STRIPE_PERIOD_COSINE_A + STRIPE_PERIOD_COSINE_B * np.cos(np.pi * t / L + STRIPE_PERIOD_COSINE_PHI)
        T_t = np.maximum(T_t, 1.0)  # avoid div by zero
        v = 2.0 * np.pi / (q * T_t)
        # Extra amplitude modulation on top of T(t) (independent period v_period_s).
        v = v * (1.0 + v_amp_frac * np.cos(2.0 * np.pi * t / v_period_s))
        v = np.maximum(v, 1e-6 * v_mean)
    else:
        v_early = v_mean * 1.5
        v_late = v_mean * 0.5
        transition = 0.45 * tmax
        width = 0.08 * tmax
        v = v_late + (v_early - v_late) * 0.5 * (1.0 - np.tanh((t - transition) / width))
    if constant_J_and_xs:
        # Constant J and x_s (no time-dependent ramp); J gives decay exp(-q²∫J dt) along delay.
        J = np.full_like(t, J_constant)
        x_s = np.full_like(t, 0.5)
        return J, v, x_s
    # Time-dependent J(t) and x_s(t) in forward t (paper SI Section 3). Both increase with t over the window → ramp.
    x_s = 0.25 + 0.45 * (1.0 - np.exp(-t / (xs_tau_frac * tmax)))
    J = J0 * (1.0 - np.exp(-t / (j_tau_frac * tmax)))
    return J, v, x_s


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def plot_ttc(
    t: np.ndarray,
    c2: np.ndarray,
    ax: plt.Axes | None = None,
    *,
    title: str = "c₂(q, t₁, t₂)",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    xlabel: str = "t₁ (s)",
    ylabel: str = "t₂ (s)",
    mirror_right_about_diagonal: bool = True,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # Optionally show left portion (t1 < t2) as mirror of right about t1=t2
    if mirror_right_about_diagonal:
        c2 = np.triu(c2) + np.tril(c2.T, k=-1)
    if vmin is None:
        vmin = np.nanmin(c2)
    if vmax is None:
        vmax = np.nanmax(c2)
    im = ax.imshow(
        c2,
        origin="lower",
        cmap=cmap,
        aspect="equal",
        extent=[t[0], t[-1], t[0], t[-1]],
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="c₂", fraction=0.046, pad=0.02)
    return ax


# ---------------------------------------------------------------------------
#  Main: build TTCs and plot
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce TTC plots from transport-coefficient analytic model."
    )
    parser.add_argument(
        "--laminar",
        action="store_true",
        default=DEFAULT_LAMINAR,
        help="Plot only laminar (homodyne) TTC",
    )
    parser.add_argument(
        "--heterodyne",
        action="store_true",
        default=DEFAULT_HETERODYNE,
        help="Plot only heterodyne (two-component) TTC",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=DEFAULT_SAVE,
        help="Save figures to PNG in current directory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=DEFAULT_NO_SHOW,
        help="Do not call plt.show() (e.g. when saving headless)",
    )
    parser.add_argument(
        "--time-range",
        type=str,
        default=None,
        metavar="SPEC",
        help="Time range: 'full' or 'start,end,stride' (default from script)",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=DEFAULT_DT_S,
        metavar="SEC",
        help="Seconds per step (default %(default)s)",
    )
    parser.add_argument(
        "--periodic-v",
        action="store_true",
        default=DEFAULT_V_PERIODIC,
        dest="periodic_v",
        help="Use periodic v(t) for heterodyne (default: %s)" % DEFAULT_V_PERIODIC,
    )
    parser.add_argument(
        "--no-periodic-v",
        action="store_false",
        dest="periodic_v",
        help="Use tanh two-regime v(t) instead of periodic",
    )
    parser.add_argument(
        "--v-period",
        type=float,
        default=DEFAULT_V_PERIOD_S,
        metavar="SEC",
        help="Period of v(t) in seconds when --periodic-v (default %(default)s)",
    )
    parser.add_argument(
        "--v-amp-frac",
        type=float,
        default=DEFAULT_V_AMP_FRAC,
        metavar="FRAC",
        help="Amplitude of v(t) oscillation as fraction of v_mean when --periodic-v (default %(default)s)",
    )
    parser.add_argument(
        "--j-constant",
        type=float,
        default=DEFAULT_J_CONSTANT,
        metavar="J",
        help="Transport coeff J (constant or J0 when time-dep). Lower = gentler antidiagonal decay (default %(default)s)",
    )
    args = parser.parse_args()
    use_periodic_v = args.periodic_v

    time_range = DEFAULT_TIME_RANGE if args.time_range is None else (
        "full" if args.time_range.strip().lower() == "full" else [int(x) for x in args.time_range.split(",")]
    )
    if time_range != "full" and len(time_range) != 3:
        raise ValueError("--time-range must be 'full' or 'start,end,stride' (three integers)")
    dt_s = args.dt_s
    t = time_axis_from_range(time_range, dt_s)
    t_max = float(t[-1]) if len(t) > 0 else 0.0

    do_both = not (args.laminar or args.heterodyne)

    if do_both:
        # Single figure with two panels
        fig, (ax_lam, ax_het) = plt.subplots(1, 2, figsize=(12, 5))
        J_lam, gammadot = example_laminar_params(t, t_max)
        c2_lam = c2_laminar(t, J_lam, gammadot, q=0.054, h=1.0, phi_deg=0.0, beta=0.5)
        J_het, v, x_s = example_heterodyne_params(
            t, t_max,
            stripe_period_s=DEFAULT_STRIPE_PERIOD_S,
            constant_J_and_xs=DEFAULT_CONSTANT_J_AND_XS,
            use_periodic_v=use_periodic_v,
            v_period_s=args.v_period,
            v_amp_frac=args.v_amp_frac,
            xs_tau_frac=DEFAULT_XS_TAU_FRAC,
            j_tau_frac=DEFAULT_J_TAU_FRAC,
            J_constant=args.j_constant,
            J0=args.j_constant,
        )
        c2_het = c2_heterodyne(t, J_het, v, x_s, q=0.054, phi_deg=0.0, beta=0.5)
        plot_ttc(t, c2_lam, ax=ax_lam, title="Laminar (homodyne) — Eq. 1", cmap="viridis")
        plot_ttc(t, c2_het, ax=ax_het, title="Heterodyne (two-component) — Eq. 2", cmap="plasma")
        plt.tight_layout()
        if args.save:
            fig.savefig("ttc_both.png", dpi=150, bbox_inches="tight")
            print("Saved ttc_both.png")
        if not args.no_show:
            plt.show()
        return

    if args.laminar:
        path = _ttc_path(POSITION_NAME, SAMPLE_ID, MASK_N)
        if not path.exists():
            raise FileNotFoundError(f"TTC array not found: {path}")
        C_real = load_ttc(path)
        C_real = apply_time_range(C_real, time_range)
        if C_real.shape[0] != len(t) or C_real.shape[1] != len(t):
            raise ValueError(f"Real TTC shape {C_real.shape} does not match time length {len(t)}")
        J, gammadot = example_laminar_params(t, t_max)
        c2_lam = c2_laminar(
            t, J, gammadot,
            q=0.054, h=1.0, phi_deg=0.0, beta=0.5,
        )
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
        plot_ttc(t, C_real, ax=ax_left, title="Data (laminar)", cmap="viridis")
        plot_ttc(t, c2_lam, ax=ax_right, title="Model (laminar) — Eq. 1", cmap="viridis")
        plt.tight_layout()
        if args.save:
            fig.savefig("ttc_laminar.png", dpi=150, bbox_inches="tight")
            print("Saved ttc_laminar.png")
        if not args.no_show:
            plt.show()
        return

    if args.heterodyne:
        path = _ttc_path(POSITION_NAME, SAMPLE_ID, MASK_N)
        if not path.exists():
            raise FileNotFoundError(f"TTC array not found: {path}")
        C_real = load_ttc(path)
        C_real = apply_time_range(C_real, time_range)
        if C_real.shape[0] != len(t) or C_real.shape[1] != len(t):
            raise ValueError(f"Real TTC shape {C_real.shape} does not match time length {len(t)}")
        J, v, x_s = example_heterodyne_params(
            t, t_max,
            stripe_period_s=DEFAULT_STRIPE_PERIOD_S,
            constant_J_and_xs=DEFAULT_CONSTANT_J_AND_XS,
            use_periodic_v=use_periodic_v,
            v_period_s=args.v_period,
            v_amp_frac=args.v_amp_frac,
            xs_tau_frac=DEFAULT_XS_TAU_FRAC,
            j_tau_frac=DEFAULT_J_TAU_FRAC,
            J_constant=args.j_constant,
            J0=args.j_constant,
        )
        c2_het = c2_heterodyne(
            t, J, v, x_s,
            q=0.054, phi_deg=0.0, beta=0.5,
        )
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
        plot_ttc(t, C_real, ax=ax_left, title="Data (heterodyne)", cmap="plasma")
        plot_ttc(t, c2_het, ax=ax_right, title="Model (heterodyne) — Eq. 2", cmap="plasma")
        plt.tight_layout()
        if args.save:
            fig.savefig("ttc_heterodyne.png", dpi=150, bbox_inches="tight")
            print("Saved ttc_heterodyne.png")
        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()
