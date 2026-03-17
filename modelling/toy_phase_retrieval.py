"""
toy_phase_retrieval.py

Minimal HIO + ER phase retrieval demo using the BCDI forward model
as the ground-truth data source.

Generates a 3D coherent diffraction pattern from a strained
nanocrystal (via bragg_forward_model), then recovers the real-space
complex object iteratively.  Plots ground-truth vs. reconstruction.

Usage:
    python modelling/toy_phase_retrieval.py
    python modelling/toy_phase_retrieval.py --grid-size 64 --hio-iters 300 --er-iters 100
"""

from __future__ import annotations

import argparse
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import matplotlib.pyplot as plt

from bragg_forward_model import (
    BCDIForwardParams, forward_model, make_support, print_summary,
)


# ------------------------------------------------------------------
#  Phase retrieval core
# ------------------------------------------------------------------
def phase_retrieval(
    magnitudes: np.ndarray,
    support: np.ndarray,
    *,
    n_hio: int = 200,
    n_er: int = 50,
    beta: float = 0.9,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run HIO followed by ER on a 3D diffraction magnitude array.

    Returns the recovered complex object and per-iteration error metric.
    """
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(magnitudes.shape) + 1j * rng.standard_normal(magnitudes.shape)

    inside = support > 0
    errors = []

    for it in range(n_hio + n_er):
        PSI = fftshift(fftn(psi))

        err = np.sqrt(np.sum((np.abs(PSI) - magnitudes) ** 2) / np.sum(magnitudes ** 2))
        errors.append(err)

        PSI = magnitudes * np.exp(1j * np.angle(PSI))
        psi_prime = ifftn(ifftshift(PSI))

        if it < n_hio:
            psi = np.where(inside, psi_prime, psi - beta * psi_prime)
        else:
            psi = np.where(inside, psi_prime, 0.0)

    return psi, np.array(errors)


# ------------------------------------------------------------------
#  Plotting
# ------------------------------------------------------------------
def plot_comparison(
    psi_true: np.ndarray,
    psi_recon: np.ndarray,
    support: np.ndarray,
    errors: np.ndarray,
    params: BCDIForwardParams,
    n_hio: int,
) -> plt.Figure:
    """Side-by-side comparison of ground truth vs reconstruction."""
    N = psi_true.shape[0]
    mid = N // 2

    amp_true = np.abs(psi_true)
    phase_true = np.angle(psi_true)
    amp_recon = np.abs(psi_recon)
    phase_recon = np.angle(psi_recon)

    phase_true = np.where(support > 0, phase_true, np.nan)
    phase_recon = np.where(support > 0, phase_recon, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 0: amplitude
    im = axes[0, 0].imshow(amp_true[mid], cmap="gray", origin="lower")
    axes[0, 0].set_title("Ground truth |ψ|")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.7)

    im = axes[0, 1].imshow(amp_recon[mid], cmap="gray", origin="lower")
    axes[0, 1].set_title("Reconstructed |ψ|")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.7)

    # Row 1: phase (inside support only)
    vmin = np.nanmin(phase_true)
    vmax = np.nanmax(phase_true)
    im = axes[1, 0].imshow(phase_true[mid], cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Ground truth phase")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.7, label="rad")

    im = axes[1, 1].imshow(phase_recon[mid], cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Reconstructed phase")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.7, label="rad")

    # Error metric
    axes[0, 2].semilogy(errors, "C0", lw=0.8)
    axes[0, 2].axvline(n_hio, color="C3", ls="--", lw=0.8, label="HIO → ER")
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("R-factor")
    axes[0, 2].set_title("Convergence")
    axes[0, 2].legend(fontsize=8)

    # Support overlay
    axes[1, 2].imshow(amp_recon[mid], cmap="gray", origin="lower")
    axes[1, 2].contour(support[mid], levels=[0.5], colors="C1", linewidths=0.8)
    axes[1, 2].set_title("Reconstruction + support")

    h, k, l = params.hkl
    strain_str = f"ε_xx={params.strain_voigt[0]:.1e}" if any(s != 0 for s in params.strain_voigt) else "no strain"
    fig.suptitle(
        f"Toy phase retrieval — ({h}{k}{l}), "
        f"{params.particle_shape} {params.particle_size_nm:.0f} nm, "
        f"{strain_str}, N={params.grid_size}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Toy BCDI phase retrieval demo.")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--particle-size", type=float, default=150.0)
    p.add_argument("--shape", choices=["sphere", "cube"], default="sphere")
    p.add_argument("--strain-xx", type=float, default=2e-3)
    p.add_argument("--hio-iters", type=int, default=200)
    p.add_argument("--er-iters", type=int, default=50)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    params = BCDIForwardParams(
        grid_size=args.grid_size,
        voxel_size_nm=5.0,
        particle_shape=args.shape,
        particle_size_nm=args.particle_size,
        strain_voigt=(args.strain_xx, 0, 0, 0, 0, 0),
    )
    print_summary(params)

    print("\n--- Forward model ---")
    intensity, q_axes = forward_model(params)
    magnitudes = np.sqrt(intensity)
    print(f"Intensity shape: {intensity.shape}")
    print(f"Max intensity:   {intensity.max():.4e}")

    support = make_support(params)

    # Build ground-truth psi for comparison
    N = params.grid_size
    voxel_A = params.voxel_size_nm * 10.0
    coords_1d = (np.arange(N) - N // 2) * voxel_A
    rz, ry, rx = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    Geps = params.G_hkl @ params.strain_tensor
    phase_true = Geps[0] * rx + Geps[1] * ry + Geps[2] * rz
    psi_true = support * np.exp(1j * phase_true)

    print(f"\n--- Phase retrieval ({args.hio_iters} HIO + {args.er_iters} ER) ---")
    psi_recon, errors = phase_retrieval(
        magnitudes, support,
        n_hio=args.hio_iters, n_er=args.er_iters,
        beta=args.beta, seed=args.seed,
    )
    print(f"Final R-factor: {errors[-1]:.6f}")

    # Phase retrieval has global phase + conjugate/inversion ambiguities;
    # pick the twin with the best phase match to the ground truth.
    inside = support > 0
    twins = [psi_recon, np.conj(psi_recon),
             psi_recon[::-1, ::-1, ::-1],
             np.conj(psi_recon[::-1, ::-1, ::-1])]
    best_corr, best_psi = -2.0, psi_recon
    for twin in twins:
        offset = np.angle(np.sum(twin[inside] * np.conj(psi_true[inside])))
        aligned = twin * np.exp(-1j * offset)
        corr = np.corrcoef(np.angle(aligned[inside]),
                           np.angle(psi_true[inside]))[0, 1]
        if corr > best_corr:
            best_corr, best_psi = corr, aligned
    psi_recon = best_psi
    print(f"Phase correlation with ground truth: {best_corr:.4f}")

    fig = plot_comparison(psi_true, psi_recon, support, errors, params, args.hio_iters)
    plt.show()


if __name__ == "__main__":
    main()
