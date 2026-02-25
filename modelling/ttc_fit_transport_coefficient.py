"""
Fit experimental TTC arrays with the transport-coefficient model (He et al. 2024/2025).

Loads a saved .npy from data/ttc_arrays/<POSITION_NAME>/<SAMPLE_ID>/ and fits
either laminar (homodyne) or heterodyne (two-component) model. Set run and time
range just below, then run: python ttc_fit_transport_coefficient.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.fft import rfft, rfftfreq

from ttc_plots_transport_coefficient import c2_laminar, c2_heterodyne


# ---------------------------------------------------------------------------
#  Run configuration (set these for your data)
# ---------------------------------------------------------------------------
SAMPLE_ID = "A073"
MASK_N = 145
POSITION_NAME = "A4"

# Time range: "full" to use the entire array, or [index_start, index_end, stride]
# Example: [0, 400, 2] uses rows/cols 0:400:2 (smaller, faster fit).
TIME_RANGE = [1000, 3000, 2] # or e.g. [0, 300, 2]

# Time axis: seconds per frame (used to build t = 0, dt_s, 2*dt_s, ... for sliced length).
DT_S = 0.5

# Fit mode: "laminar" or "heterodyne"
MODE = "heterodyne"

# Heterodyne: approximate stripe period (s) from the data; used to seed v0 = 2π/(q*period).
# Set to None to use FFT-based estimate instead. With period ≈ 175 s, v ≈ 0.67 Å/s.
STRIPE_PERIOD_S = 175.0

# If True, use constant J(t) and constant x_s(t) so the model has no top-left→bottom-right
# intensity gradient (stripes only, no mean drift with absolute time).
CONSTANT_J_AND_XS = True


# ---------------------------------------------------------------------------
#  Paths and load
# ---------------------------------------------------------------------------
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


def estimate_v_from_stripes(
    C: np.ndarray,
    dt_s: float,
    q: float = 0.054,
) -> float:
    """
    Estimate velocity (Å/s) from stripe period in the TTC.
    Stripes come from cos(q v τ); we take lineouts along constant delay τ = t2 - t1,
    average over a few diagonals, FFT, and get peak frequency f → v = 2π f / q.
    """
    n = C.shape[0]
    if n < 20:
        return 5.0
    # Lineout along delay: for delay index k, average C[i, i+k] over valid i
    max_k = n - 1
    lineout = np.zeros(max_k + 1)
    for k in range(max_k + 1):
        lineout[k] = np.nanmean(np.diag(C, k))
    lineout = np.nan_to_num(lineout, nan=1.0)
    # τ in seconds
    tau = np.arange(len(lineout), dtype=float) * dt_s
    # FFT (real signal)
    n_fft = len(lineout)
    fft_vals = rfft(lineout - np.mean(lineout))
    freqs = rfftfreq(n_fft, d=dt_s)
    power = np.abs(fft_vals) ** 2
    # Skip DC and very high freq; find peak in 0.001–0.5 Hz or similar
    valid = (freqs > 0.001) & (freqs < 0.5)
    if not np.any(valid):
        return 5.0
    idx_peak = np.argmax(power[valid])
    f_peak = freqs[valid][idx_peak]
    # ω = q*v  →  v = 2π f / q
    v_guess = 2.0 * np.pi * f_peak / q
    return max(0.2, min(50.0, float(v_guess)))


# ---------------------------------------------------------------------------
#  Parametrization: vector of params -> J, gammadot or J, v, x_s on grid t
#  Heterodyne forms match ttc_plots_transport_coefficient.py (forward-time).
# ---------------------------------------------------------------------------
def params_to_laminar(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """p = [J0, tau_J, alpha_J, g0, tau_g, beta]. Returns (J, gammadot)."""
    J0, tau_J, alpha_J, g0, tau_g, beta = p
    tau_J = max(tau_J, 1e-6)
    tau_g = max(tau_g, 1e-6)
    J = J0 * (1.0 + t / tau_J) ** (-alpha_J)
    gammadot = g0 * (1.0 + t / tau_g) ** (-2.0 / 3.0)
    return J, gammadot


def params_to_heterodyne(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same functional forms as ttc_plots_transport_coefficient.example_heterodyne_params
    (forward time t): J(t) = J0*(1 - exp(-t/tau_J)), x_s(t) = 0.25 + 0.45*(1 - exp(-t/tau_xs)),
    v(t) = v_mean*(1 + v_amp_frac*cos(2π t/v_period_s)), clipped to stay positive.
    p = [J0, j_tau_frac, v_mean, v_amp_frac, v_period_s, xs_tau_frac, beta]. Returns (J, v, x_s).
    """
    J0, j_tau_frac, v_mean, v_amp_frac, v_period_s, xs_tau_frac, beta = p
    t_max = float(t[-1]) if len(t) > 0 else 1.0
    j_tau_frac = max(j_tau_frac, 1e-4)
    xs_tau_frac = max(xs_tau_frac, 1e-4)
    v_period_s = max(v_period_s, 1e-3)
    v_mean = max(v_mean, 0.0)
    v_amp_frac = np.clip(v_amp_frac, 0.0, 0.99)

    tau_J = j_tau_frac * t_max
    tau_xs = xs_tau_frac * t_max
    J = J0 * (1.0 - np.exp(-t / tau_J))
    x_s = 0.25 + 0.45 * (1.0 - np.exp(-t / tau_xs))
    x_s = np.clip(x_s, 0.0, 1.0)
    v = v_mean * (1.0 + v_amp_frac * np.cos(2.0 * np.pi * t / v_period_s))
    v = np.maximum(v, 1e-6 * v_mean)
    return J, v, x_s


# ---------------------------------------------------------------------------
#  Objective and fit
# ---------------------------------------------------------------------------
def _sse_laminar(
    p: np.ndarray,
    t: np.ndarray,
    C_data: np.ndarray,
    q: float,
    h: float,
    phi_deg: float,
    use_upper_triangle: bool,
) -> float:
    J, gammadot = params_to_laminar(t, p)
    beta = p[5]
    beta = np.clip(beta, 0.01, 1.0)
    C_mod = c2_laminar(t, J, gammadot, q=q, h=h, phi_deg=phi_deg, beta=beta)
    if use_upper_triangle:
        iu = np.triu_indices_from(C_data, k=1)
        return float(np.nansum((C_data[iu] - C_mod[iu]) ** 2))
    return float(np.nansum((C_data - C_mod) ** 2))


def _sse_heterodyne(
    p: np.ndarray,
    t: np.ndarray,
    C_data: np.ndarray,
    q: float,
    phi_deg: float,
    use_upper_triangle: bool,
) -> float:
    J, v, x_s = params_to_heterodyne(t, p)
    beta = p[6]
    beta = np.clip(beta, 0.01, 1.0)
    C_mod = c2_heterodyne(t, J, v, x_s, q=q, phi_deg=phi_deg, beta=beta)
    if use_upper_triangle:
        iu = np.triu_indices_from(C_data, k=1)
        return float(np.nansum((C_data[iu] - C_mod[iu]) ** 2))
    return float(np.nansum((C_data - C_mod) ** 2))


def _sse_heterodyne_flat(
    p: np.ndarray,
    t: np.ndarray,
    C_data: np.ndarray,
    q: float,
    phi_deg: float,
    use_upper_triangle: bool,
) -> float:
    """4-param: p = [J0, v0, x_s_const, beta]. Constant J and x_s → no gradient, stripes only."""
    J0, v0, x_s_const, beta = p
    J = np.full_like(t, max(J0, 1e-6))
    v = np.full_like(t, max(v0, 0.0))
    x_s = np.full_like(t, np.clip(x_s_const, 0.0, 1.0))
    beta = np.clip(beta, 0.01, 1.0)
    C_mod = c2_heterodyne(t, J, v, x_s, q=q, phi_deg=phi_deg, beta=beta)
    if use_upper_triangle:
        iu = np.triu_indices_from(C_data, k=1)
        return float(np.nansum((C_data[iu] - C_mod[iu]) ** 2))
    return float(np.nansum((C_data - C_mod) ** 2))


def fit_laminar(
    t: np.ndarray,
    C: np.ndarray,
    *,
    q: float = 0.054,
    h: float = 1.0,
    phi_deg: float = 0.0,
    use_upper_triangle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (best_params, C_model)."""
    n = len(t)
    t_max = float(t[-1] - t[0]) if n > 1 else 1.0
    # p = [J0, tau_J, alpha_J, g0, tau_g, beta]
    p0 = np.array([1.0, t_max / 2.0, 1.2, 0.01, t_max / 2.0, 0.5])
    bounds = [
        (1e-4, 50.0),
        (1e-3, 1e4),
        (0.1, 3.0),
        (1e-6, 1.0),
        (1e-3, 1e4),
        (0.01, 1.0),
    ]
    C = np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=1.0)
    res = minimize(
        _sse_laminar,
        p0,
        args=(t, C, q, h, phi_deg, use_upper_triangle),
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=300),
    )
    p_best = res.x
    J, gammadot = params_to_laminar(t, p_best)
    beta = np.clip(p_best[5], 0.01, 1.0)
    C_mod = c2_laminar(t, J, gammadot, q=q, h=h, phi_deg=phi_deg, beta=beta)
    return p_best, C_mod


def fit_heterodyne(
    t: np.ndarray,
    C: np.ndarray,
    *,
    q: float = 0.054,
    phi_deg: float = 0.0,
    use_upper_triangle: bool = True,
    dt_s: float | None = None,
    v_min: float = 0.2,
    stripe_period_s: float | None = None,
    constant_J_and_xs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (best_params, C_model). If constant_J_and_xs, fit 4 params (J0, v0, x_s, beta) for stripes only, no gradient."""
    n = len(t)
    t_max = float(t[-1] - t[0]) if n > 1 else 1.0
    dt = (t[1] - t[0]) if n > 1 else 1.0
    if dt_s is None:
        dt_s = dt
    C_clean = np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=1.0)
    if stripe_period_s is not None and stripe_period_s > 0:
        v_guess = 2.0 * np.pi / (q * float(stripe_period_s))
        v_guess = max(v_min, min(50.0, v_guess))
    else:
        v_guess = estimate_v_from_stripes(C_clean, dt_s, q=q)

    if constant_J_and_xs:
        # 4 params: J0, v0, x_s_const, beta → no gradient, stripes only
        p0 = np.array([0.5, v_guess, 0.5, 0.5])
        bounds = [(1e-4, 20.0), (v_min, 50.0), (0.0, 1.0), (0.01, 1.0)]
        res = minimize(
            _sse_heterodyne_flat,
            p0,
            args=(t, C_clean, q, phi_deg, use_upper_triangle),
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=300),
        )
        p_best = res.x
        J = np.full_like(t, max(p_best[0], 1e-6))
        v = np.full_like(t, max(p_best[1], 0.0))
        x_s = np.full_like(t, np.clip(p_best[2], 0.0, 1.0))
        beta = np.clip(p_best[3], 0.01, 1.0)
        C_mod = c2_heterodyne(t, J, v, x_s, q=q, phi_deg=phi_deg, beta=beta)
        return p_best, C_mod

    # Full 7-param fit: [J0, j_tau_frac, v_mean, v_amp_frac, v_period_s, xs_tau_frac, beta]
    # Same forms as ttc_plots_transport_coefficient (forward-time J, x_s; periodic v).
    p0 = np.array([1.5, 0.6, v_guess, 0.2, 150.0, 0.25, 0.5])
    bounds = [
        (1e-4, 20.0),
        (0.01, 2.0),   # j_tau_frac (fraction of t_max)
        (v_min, 50.0),
        (0.0, 0.99),   # v_amp_frac
        (1.0, 1e4),    # v_period_s [s]
        (0.01, 2.0),   # xs_tau_frac
        (0.01, 1.0),
    ]
    res = minimize(
        _sse_heterodyne,
        p0,
        args=(t, C_clean, q, phi_deg, use_upper_triangle),
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=300),
    )
    p_best = res.x
    J, v, x_s = params_to_heterodyne(t, p_best)
    beta = np.clip(p_best[6], 0.01, 1.0)
    C_mod = c2_heterodyne(t, J, v, x_s, q=q, phi_deg=phi_deg, beta=beta)
    return p_best, C_mod


# ---------------------------------------------------------------------------
#  Plot data vs model
# ---------------------------------------------------------------------------
def plot_data_vs_model(
    t: np.ndarray,
    C_data: np.ndarray,
    C_model: np.ndarray,
    *,
    clip_pct: float = 99.9,
    title_prefix: str = "",
) -> None:
    vmin = np.nanpercentile(C_data, 0)
    vmax = np.nanpercentile(C_data, clip_pct)
    vmax = max(vmax, vmin + 1e-6)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    ax0.imshow(
        np.clip(C_data, vmin, vmax),
        origin="lower",
        cmap="plasma",
        aspect="equal",
        extent=[t[0], t[-1], t[0], t[-1]],
        interpolation="nearest",
    )
    ax0.set_title(f"{title_prefix}Data TTC")
    ax0.set_xlabel("t₁ (s)")
    ax0.set_ylabel("t₂ (s)")
    ax1.imshow(
        np.clip(C_model, vmin, vmax),
        origin="lower",
        cmap="plasma",
        aspect="equal",
        extent=[t[0], t[-1], t[0], t[-1]],
        interpolation="nearest",
    )
    ax1.set_title(f"{title_prefix}Model TTC (fitted)")
    ax1.set_xlabel("t₁ (s)")
    ax1.set_ylabel("t₂ (s)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    path = _ttc_path(POSITION_NAME, SAMPLE_ID, MASK_N)
    if not path.exists():
        raise FileNotFoundError(f"TTC array not found: {path}")

    C = load_ttc(path)
    C = apply_time_range(C, TIME_RANGE)
    n = C.shape[0]
    if n != C.shape[1]:
        raise ValueError(f"TTC must be square after slice, got {C.shape}")

    t = np.linspace(0.0, (n - 1) * DT_S, n)

    if MODE == "laminar":
        p_best, C_mod = fit_laminar(t, C, q=0.054, h=1.0, phi_deg=0.0)
        print("Laminar fit params: J0=%.4f tau_J=%.2f alpha_J=%.4f g0=%.6f tau_g=%.2f beta=%.4f" % tuple(p_best))
    elif MODE == "heterodyne":
        p_best, C_mod = fit_heterodyne(
            t, C,
            q=0.054, phi_deg=0.0, dt_s=DT_S,
            stripe_period_s=STRIPE_PERIOD_S,
            constant_J_and_xs=CONSTANT_J_AND_XS,
        )
        if CONSTANT_J_AND_XS:
            print("Heterodyne (flat) fit params: J0=%.4f v0=%.4f x_s=%.4f beta=%.4f" % tuple(p_best))
        else:
            print(
                "Heterodyne fit params: J0=%.4f j_tau_frac=%.4f v_mean=%.4f v_amp_frac=%.4f v_period_s=%.2f xs_tau_frac=%.4f beta=%.4f"
                % tuple(p_best)
            )
    else:
        raise ValueError('MODE must be "laminar" or "heterodyne"')

    sse = float(np.nansum((C - C_mod) ** 2))
    print("SSE =", sse)

    plot_data_vs_model(
        t, C, C_mod,
        title_prefix=f"{SAMPLE_ID} M{MASK_N} {MODE} ",
    )
