"""
APS 08-ID-E XPCS analysis (2025-1006): inspection, Bragg peak metrics, q/φ maps.

Usage:
  python analysis_for_aps_08-ide-2025-1006.py <command> [--file-id ID] [--filename PATH] [options]
  python analysis_for_aps_08-ide-2025-1006.py --help
  python analysis_for_aps_08-ide-2025-1006.py <command> --help

Tab completion (optional, pip only): use the wrapper so Tab works (completion does not run for "python script.py").
  pip install -r emilio_scripts/python_scripts/requirements-argcomplete.txt
  activate-global-python-argcomplete   # one-time, or: activate-global-python-argcomplete --dest=- >> ~/.zshrc
  source emilio_scripts/python_scripts/argcomplete-setup.sh   # add to ~/.zshrc
  Then run: aps_analysis <TAB>  (from that dir, or add the dir to PATH)
"""
import argparse
import os
from pathlib import Path

try:
    import argcomplete
except ImportError:
    argcomplete = None

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

try:
    from scipy import stats as scipy_stats
    from scipy import optimize as scipy_optimize
    from scipy.ndimage import label as ndi_label
except ImportError:
    scipy_stats = None
    scipy_optimize = None
    ndi_label = None

# from source_functions import *
# from xpcs import *
# from sims import *


# ---------------------------------------------------------------------------
# HDF5 inspection helpers
# ---------------------------------------------------------------------------

def print_h5_structure(name, obj):
    print(name)


def explore(name, obj):
    indent = "  " * (name.count("/") - 1)
    if isinstance(obj, h5py.Group):
        print(f"{indent}{name}/  (Group)")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}  (Dataset)  shape={obj.shape}, dtype={obj.dtype}")

def h5_file_inspector(filename):
    with h5py.File(filename, "r") as f:
        f.visititems(explore)


def _symmetrize_ttc(C):
    """Return symmetrized two-time matrix: C + C.T - diag(diag(C))."""
    C = np.asarray(C, dtype=np.float64)
    return C + C.T - np.diag(np.diag(C))


def _clip_ttc(C, p_hi=99.9):
    """Clip TTC to [0, p_hi] percentile for display."""
    lo, hi = np.percentile(C, [0.0, float(p_hi)])
    return np.clip(C, lo, hi)


def _compute_mask_integrated_intensities(roi_map, scattering_2d, n_masks=300, label_offset=0):
    """Return list of integrated intensities. Labels used: label_offset .. label_offset+n_masks-1 (e.g. 0..299 or 1..300)."""
    scat = np.asarray(scattering_2d)
    if scat.ndim == 3:
        scat = scat[0, :, :]
    return [
        float(np.nansum(scat[roi_map == (i + label_offset)]))
        for i in range(n_masks)
    ]


# ---------------------------------------------------------------------------
# Simple plotters (g2, TTC, intensity, bins)
# ---------------------------------------------------------------------------

def g2_plotter(filename):
    with h5py.File(filename, "r") as f:
        g2 = f["xpcs/twotime/normalized_g2"][...]

    x = np.arange(len(g2[:, 100]))

    plt.figure()
    # plt.errorbar(delay[0, :], G2_result, yerr=G2_error, fmt='none', ecolor='b', capsize=2)
    plt.semilogx(x, g2[:, 194], 'b.')
    plt.title('g2 autocorrelation with pixelwise normalisation')
    plt.ylabel('g2(q,tau)')
    plt.xlabel('Delay Time, tau (s)')
    # plt.ylim([0, 1.5])

    plt.show()

def ttc_plotter(filename):
    with h5py.File(filename, "r") as f:
        C = f["xpcs/twotime/correlation_map/c2_00194"][...]
    C = _symmetrize_ttc(C)
    C = C - np.min(C)
    C = C / np.max(C)
    # C_extent = [0, frameSpacing * det.shape[0], 0, frameSpacing * det.shape[0]]
    # C_extent = [0, det.shape[0], 0, det.shape[0]]

    plt.rcParams.update({'font.size': 24})

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_ylabel('t2 (s)')
    ax.set_xlabel('t1 (s)')
    ax.set_ylabel('Frame number')
    ax.set_xlabel('Frame number')
    # ax.set_title(f'Ring {0}')
    # ax.set_title('Whole Mask')
    im = ax.imshow(C, origin="lower", cmap='plasma')
    cbar = fig.colorbar(im, ax=ax)
    custom_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cbar.set_ticks(custom_ticks)

    # diag_vals = np.diag(C)
    # x_min, x_max, y_min, y_max = C_extent
    # n = C.shape[0]  # assuming square matrix
    # t_axis = np.linspace(x_min, x_max, n)
    #
    # plt.figure(figsize=(8, 4))
    # plt.plot(np.arange(0, len(t_axis), 1), 1 - diag_vals, marker='.', linestyle='-', alpha=0.8, color='C1')
    # plt.xlabel('t (s)')
    # plt.ylabel('C(t,t)')
    # plt.title('Diagonal of C (t1 = t2)')
    # plt.grid(True)
    # plt.tight_layout()

    plt.show()

def intensity_vs_time(filename):
    with h5py.File(filename, "r") as f:
        data = f["xpcs/spatial_mean/intensity_vs_time"][...]

    print(data.shape)

    plt.figure()
    plt.plot(data[0, :], data[1, :])
    plt.show()

def static_vs_dynamic_bins(filename):
    with h5py.File(filename, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]

    scattering_2d_reshape = scattering_2d[0, :, :] if scattering_2d.ndim == 3 else scattering_2d
    individual_mask_intensity = _compute_mask_integrated_intensities(
        dynamic_roi_map, scattering_2d_reshape, n_masks=300, label_offset=1
    )

    print(dynamic_roi_map.shape)
    print(scattering_2d.shape)
    print(scattering_2d_reshape.shape)

    plt.figure()
    plt.semilogy(np.arange(1, 301, 1), individual_mask_intensity)
    plt.xlabel('mask number')
    plt.ylabel('integrated intensity')

    plt.figure()
    plt.imshow(dynamic_roi_map)

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    I = scattering_2d_reshape.astype(float)

    plt.imshow(I,
        origin="upper",
        cmap=cmap,
        norm=LogNorm(vmin=0.1, vmax=np.max(I)),
        interpolation="nearest",
    )
    # ax0.set_facecolor("black")

    # np.savetxt('dynamic_roi_map.txt', dynamic_roi_map)
    # print(np.shape(dynamic_roi_map))


    plt.show()

def combined_plot(filename):
    with h5py.File(filename, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]
        q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]

    run_name = os.path.basename(filename).split("_")[0]
    scattering_2d_reshape = scattering_2d[0, :, :] if scattering_2d.ndim == 3 else scattering_2d
    individual_mask_intensity = _compute_mask_integrated_intensities(
        dynamic_roi_map, scattering_2d_reshape, n_masks=300, label_offset=0
    )

    print('q:', q)
    print('phi:', phi)

    i = np.argmax(individual_mask_intensity)
    # idxs = [0, 1, -1, -31, -30, -29, 29, 30, 31] + i
    idxs = [-29, 1, 31, -30, 0, 30, -31, -1, 29] + i

    masks = dynamic_roi_map.copy()
    combined_mask = np.isin(masks, idxs).astype(int)

    im = dynamic_roi_map.copy()
    im[~np.isin(im, idxs)] = 0

    plt.figure()
    plt.imshow(im)

    plt.figure()
    x = np.arange(g2.shape[0])
    for midx in idxs:
        if 0 <= midx < g2.shape[1]:
            plt.semilogx(x, g2[:, midx], label=f"M{midx}, q={q[midx // 30]:.3f}, φ={phi[midx % 30]:.3f}")
    plt.title('g2 autocorrelation for experiment ' + run_name)
    plt.ylabel('g2(q,tau)')
    plt.xlabel('Delay Time, tau')
    plt.legend()

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    for i, ax in enumerate(axes.flat):
        path = f"xpcs/twotime/correlation_map/c2_00{idxs[i]:03d}"
        with h5py.File(filename, "r") as f:
            C = f[path][...]
        C = _symmetrize_ttc(C)
        C = _clip_ttc(C, 99.9)
        # C = (C_clip - lo) / (hi - lo)
        # ax.set_title(f"M{idxs[i]}")
        ax.axis("off")
        # ax.set_ylabel('Frame number')
        # ax.set_xlabel('Frame number')
        im = ax.imshow(C, origin="lower", cmap='plasma')
        label = (
            f"M{idxs[i]}\n"
            f"min {np.min(C):.2f}\n"
            f"max {np.max(C):.2f}"
        )
        ax.text(
            0.05, 0.95, label,
            transform=ax.transAxes,  # axes-relative coordinates
            ha="left", va="top",
            fontsize=12,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="black",
                alpha=0.6,
                edgecolor="none"
            )
        )
        print(path)

    plt.tight_layout()

    plt.figure()

    I = scattering_2d_reshape.astype(float)
    I[combined_mask == 1] *= 10

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")  # or "navy", etc.
    cmap.set_bad("black")  # for NaN/inf

    ys, xs = np.where(combined_mask == 1)
    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))
    half = 200
    ymin = max(cy - half, 0)
    ymax = min(cy + half, I.shape[0])
    xmin = max(cx - half, 0)
    xmax = min(cx + half, I.shape[1])
    img_crop = I[ymin:ymax, xmin:xmax]
    mask_crop = combined_mask[ymin:ymax, xmin:xmax]

    plt.imshow(img_crop, origin="lower", cmap=cmap, norm=LogNorm(vmin=0.1, vmax=I.max()))
    plt.colorbar()

    # plt.figure()
    # plt.imshow(combined_mask)

    print(run_name)
    print(filename)

    plt.show()


# ---------------------------------------------------------------------------
# Google Sheets / OAuth (optional)
# ---------------------------------------------------------------------------

def oauth_test():
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ('/Users/emilioescauriza/Documents/repos/006_APS_8IDE/emilio_scripts/python_scripts/client_secret_'
                  '180145739842-0ug37lsh4qltki62e8te8bqkde9u25jb.apps.googleusercontent.com.json')

    def get_creds():
        creds = None
        if Path("token.json").exists():
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials, SCOPES
                )
                creds = flow.run_local_server(port=0)
            Path("token.json").write_text(creds.to_json())

        return creds

    creds = get_creds()
    gc = gspread.authorize(creds)

    # Paste your spreadsheet ID here
    sh = gc.open_by_key("1OAA7H4I3cgas32aSZkrLB8TOKHymMAv2uk_0eTywWcQ")
    print("Opened spreadsheet:", sh.title)


def image_upload(fig, target_cell="AF142", upload_name="matplotlib_output.png",
                 tab_name="IPA NBH",
                 spreadsheet_id="1OAA7H4I3cgas32aSZkrLB8TOKHymMAv2uk_0eTywWcQ",
                 token_path="token.json",
                 creds_path="client_secret_180145739842-0ug37lsh4qltki62e8te8bqkde9u25jb.apps.googleusercontent.com.json"):

    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    def get_creds():
        creds = None
        if Path(token_path).exists():
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0)
            Path(token_path).write_text(creds.to_json())
        return creds

    creds = get_creds()

    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(tab_name)

    cols, rows = find_rows_with_position(ws, "A5")

    print(cols)

    for cell in rows_to_cells(rows, "AF"):
        print(cell)
        # image_upload(fig, target_cell=cell, upload_name=f"A5_{cell}.png")


    # drive = build("drive", "v3", credentials=creds)
    #
    # buf = BytesIO()
    # try:
    #     fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    #     buf.seek(0)
    #
    #     media = MediaIoBaseUpload(buf, mimetype="image/png", resumable=False)
    #     created = drive.files().create(
    #         body={"name": upload_name},
    #         media_body=media,
    #         fields="id"
    #     ).execute()
    #
    #     file_id = created["id"]
    #
    #     drive.permissions().create(
    #         fileId=file_id,
    #         body={"type": "anyone", "role": "reader"},
    #     ).execute()
    #
    #     image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
    #     formula = f'=IMAGE("{image_url}", 4, 180, 320)'
    #
    #     ws.update(target_cell, [[formula]], value_input_option="USER_ENTERED")
    #     print(f"Inserted image into {ws.title} cell {target_cell}")
    #
    # finally:
    #     buf.close()

def figure_upload():

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 4, 2])
    image_upload(fig, target_cell="AF142", upload_name="run123_overview.png")
    plt.close(fig)

def q_spacing_inspector(filename):
    with h5py.File(filename, "r") as f:
        q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]

    print('q:', q)
    print('phi:', phi)

    for i in range(9):
        print(q[i + 1] - q[i])

    for j in range(29):
        print(phi[j + 1] - phi[j])

# ---------------------------------------------------------------------------
# Integrated intensities
# ---------------------------------------------------------------------------

def integrated_intensities_inspector(filename):
    with h5py.File(filename, "r") as f:
        # dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        # scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        # ttc = f["xpcs/twotime/correlation_map/c2_00194"][...]
        # g2 = f["xpcs/twotime/normalized_g2"][...]
        # q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        # phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]
        integrated_intensities = f["xpcs/temporal_mean/scattering_1d"][...]
        integrated_intensities_segments = f["xpcs/temporal_mean/scattering_1d_segments"][...]
        q = f["xpcs/qmap/static_v_list_dim0"][...]
        phi = f["xpcs/qmap/static_v_list_dim1"][...]


        print(np.shape(integrated_intensities))
        print(np.shape(integrated_intensities_segments))
        print(integrated_intensities)

        np.savetxt("integrated_intensities.txt", integrated_intensities)
        np.savetxt("integrated_intensities_segments.txt", integrated_intensities_segments)

        print("q:", np.shape(q), q)
        print("phi:", np.shape(phi), phi)

        plt.figure()
        plt.plot(integrated_intensities[0])

        plt.figure()
        plt.plot(integrated_intensities_segments[0])
        plt.plot(integrated_intensities_segments[1])
        plt.plot(integrated_intensities_segments[2])


        # plt.ylim([0, 1])
        plt.show()

def integrated_intensities_plot(h5_file: str | Path):
    """
    Plots scattering_1d (mean) and scattering_1d_segments (10 time segments).

    Assumes flattening order where phi is the fast axis:
        flat index = iq * nphi + iphi
    so that reshape -> (nq, nphi).
    """
    h5_file = str(h5_file)

    with h5py.File(h5_file, "r") as f:
        I1d = f["xpcs/temporal_mean/scattering_1d"][...]
        Iseg = f["xpcs/temporal_mean/scattering_1d_segments"][...]
        q = f["xpcs/qmap/static_v_list_dim0"][...]
        phi = f["xpcs/qmap/static_v_list_dim1"][...]

    # ---- basic sanity ----
    I1d = np.asarray(I1d)
    Iseg = np.asarray(Iseg)
    q = np.asarray(q)
    phi = np.asarray(phi)

    if I1d.ndim != 2 or I1d.shape[0] != 1:
        raise ValueError(f"Expected scattering_1d shape (1, 3600), got {I1d.shape}")
    if Iseg.ndim != 2 or Iseg.shape[1] != I1d.shape[1]:
        raise ValueError(f"Expected scattering_1d_segments shape (10, 3600), got {Iseg.shape}")

    nq = int(q.size)
    nphi = int(phi.size)
    if nq * nphi != int(I1d.shape[1]):
        raise ValueError(
            f"q.size * phi.size = {nq}*{nphi}={nq*nphi} does not match scattering_1d length {I1d.shape[1]}"
        )

    # ---- reshape to (q, phi) ----
    I_mean_qphi = I1d[0].reshape(nq, nphi)
    I_seg_qphi = Iseg.reshape(Iseg.shape[0], nq, nphi)  # (nseg, nq, nphi)

    # Choose a representative phi index: closest to 0 degrees
    iphi0 = int(np.argmin(np.abs(phi - 0.0)))

    # ---- derived summaries ----
    # phi-averaged intensity vs q for each segment
    Iseg_q = I_seg_qphi.mean(axis=2)          # (nseg, nq)
    Imean_q = I_mean_qphi.mean(axis=1)        # (nq,)

    # variability over time segments at each (q,phi)
    I_std_qphi = I_seg_qphi.std(axis=0)       # (nq, nphi)
    I_relstd_qphi = I_std_qphi / np.maximum(I_mean_qphi, 1e-12)

    # ---- plotting ----
    fig = plt.figure(figsize=(14.5, 9.5))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.28)

    # (A) Mean map in (q, phi)
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        I_mean_qphi,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[phi.min(), phi.max(), q.min(), q.max()],
    )
    ax0.set_title("Mean scattering_1d reshaped to (q, φ)")
    ax0.set_xlabel("φ (deg)")
    ax0.set_ylabel("q (Å$^{-1}$)")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03, label="Intensity (a.u.)")

    # (B) Segment evolution as (segment index, q) for φ≈0 slice
    ax1 = fig.add_subplot(gs[0, 1])
    seg_vs_q_phi0 = I_seg_qphi[:, :, iphi0]  # (nseg, nq)
    im1 = ax1.imshow(
        seg_vs_q_phi0,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[q.min(), q.max(), 0, seg_vs_q_phi0.shape[0] - 1],
    )
    ax1.set_title(f"Segments vs q at φ≈{phi[iphi0]:.3f}° (closest to 0°)")
    ax1.set_xlabel("q (Å$^{-1}$)")
    ax1.set_ylabel("segment index")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03, label="Intensity (a.u.)")

    # (C) Line plot: φ-averaged intensity vs q for each segment + mean
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(q, Imean_q, lw=2.5, label="mean (φ-avg)")
    for s in range(Iseg_q.shape[0]):
        ax2.plot(q, Iseg_q[s], lw=1.2, alpha=0.8, label=f"seg {s}" if s < 4 else None)
    ax2.set_title("φ-averaged intensity vs q (all segments)")
    ax2.set_xlabel("q (Å$^{-1}$)")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=9)

    # (D) Relative variability map (std/mean) in (q, phi)
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(
        I_relstd_qphi,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[phi.min(), phi.max(), q.min(), q.max()],
    )
    ax3.set_title("Temporal variability: std(seg)/mean  (q, φ)")
    ax3.set_xlabel("φ (deg)")
    ax3.set_ylabel("q (Å$^{-1}$)")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03, label="Relative std")

    fig.suptitle(f"Integrated intensity diagnostics\n{h5_file}", y=0.98, fontsize=12)
    plt.show()


# ---------------------------------------------------------------------------
# Bragg peak centre and shape metrics
# ---------------------------------------------------------------------------

def find_bragg_peak_center_from_scattering_2d_with_overlay_function(
    h5_file: str | Path,
    *,
    dataset_key: str = "xpcs/temporal_mean/scattering_2d",
    use_first_frame_if_3d: bool = True,
    smooth_sigma_px: float | None = 1.0,
    bright_percentile: float = 99.7,
    weight_mode: str = "log",  # "linear" | "sqrt" | "log"
    figsize: tuple[float, float] = (7.2, 6.4),
) -> tuple[tuple[float, float], dict]:
    """
    Option 1 (robust bright-region centroid) + ALWAYS makes an overlay plot.

    Steps
    -----
    1) Load scattering_2d (use first frame if it's (1,H,W))
    2) Light smoothing (optional)
    3) Threshold at bright_percentile to define a "bright region"
    4) Compute weighted centroid of that region
    5) Plot: scattering_2d with mask outline + centroid marker

    Returns
    -------
    (cy, cx) : (float, float)
        Estimated centre (row, col) in pixel coordinates.
    info : dict
        Debug info: threshold, n_pixels_used, etc.
    """
    h5_file = Path(h5_file)

    with h5py.File(h5_file, "r") as f:
        scat = f[dataset_key][...]

    scat = np.asarray(scat)
    if scat.ndim == 3 and use_first_frame_if_3d:
        scat = scat[0, :, :]
    if scat.ndim != 2:
        raise ValueError(f"{dataset_key} must be 2D (or 3D with first-frame), got {scat.shape}")

    img = scat.astype(np.float64, copy=False)

    # Optional smoothing (very light) to stabilize centroid under hot pixels
    if smooth_sigma_px is not None and smooth_sigma_px > 0:
        try:
            from scipy.ndimage import gaussian_filter
            img_s = gaussian_filter(img, sigma=float(smooth_sigma_px))
        except Exception:
            img_s = img
    else:
        img_s = img

    # Bright-region mask
    p = float(np.clip(bright_percentile, 0.0, 100.0))
    thr = float(np.nanpercentile(img_s, p))
    mask = np.isfinite(img_s) & (img_s >= thr)
    n = int(mask.sum())

    # If too few pixels, relax threshold slightly
    if n < 10:
        p2 = max(90.0, p - 5.0)
        thr = float(np.nanpercentile(img_s, p2))
        mask = np.isfinite(img_s) & (img_s >= thr)
        n = int(mask.sum())

    if n < 10:
        raise ValueError(
            f"Bright-region mask too small (n={n}). "
            f"Lower bright_percentile (currently {bright_percentile})."
        )

    yy, xx = np.nonzero(mask)
    vals = img_s[yy, xx]

    # Weights (to reduce dominance of extreme skew / hot pixels)
    if weight_mode == "linear":
        w = np.clip(vals, 0.0, np.inf)
    elif weight_mode == "sqrt":
        w = np.sqrt(np.clip(vals, 0.0, np.inf))
    elif weight_mode == "log":
        w = np.log1p(np.clip(vals, 0.0, np.inf))
    else:
        raise ValueError("weight_mode must be one of: 'linear', 'sqrt', 'log'")

    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0:
        raise ValueError("Non-positive or non-finite weight sum, cannot compute centroid")

    cy = float(np.sum(yy * w) / wsum)
    cx = float(np.sum(xx * w) / wsum)

    info = {
        "h5_file": str(h5_file),
        "dataset_key": dataset_key,
        "img_shape": tuple(img.shape),
        "smooth_sigma_px": smooth_sigma_px,
        "bright_percentile": float(bright_percentile),
        "threshold_value": float(thr),
        "n_pixels_used": n,
        "weight_mode": weight_mode,
        "centroid_cy_cx": (cy, cx),
    }


    # Overlay plot (always)

    # Use log1p for display to handle heavy skew safely (works with zeros)
    disp = np.log1p(np.clip(img, 0.0, np.inf))

    # Robust display limits
    vmin = float(np.nanpercentile(disp, 1.0))
    vmax = float(np.nanpercentile(disp, 99.8))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = None, None

    img = np.asarray(disp, dtype=np.float64)
    img = np.where(img > 0, img, np.nan)

    vmin = np.nanpercentile(img, 1.0)
    vmax = np.nanpercentile(img, 99.999)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        img,
        origin="upper",
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    # Mask outline
    ax.contour(mask.astype(np.float32), levels=[0.5], linewidths=1.2, colors="cyan")

    # Centroid marker
    ax.plot(cx, cy, marker="x", markersize=10, mew=2.2)

    ax.set_title(
        f"Bragg peak centre (bright-region centroid)\n"
        f"p={bright_percentile:.2f}, n={n}, weight={weight_mode}, σ={smooth_sigma_px}"
    )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("log1p(scattering_2d)")

    plt.tight_layout()
    plt.show()

    return (cy, cx), info


def _ensure_2d(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D (or (1,H,W)) scattering_2d, got shape {img.shape}")
    return img


def _match_axes(img2d, q_vals, phi_vals):
    """
    Returns (img, q, phi) such that:
      img.shape == (len(phi), len(q))  i.e. axis0=phi, axis1=q
    If user provided transposed vectors, we transpose img.
    """
    q_vals = np.asarray(q_vals).ravel()
    phi_vals = np.asarray(phi_vals).ravel()

    H, W = img2d.shape
    if (H, W) == (phi_vals.size, q_vals.size):
        return img2d, q_vals, phi_vals
    if (H, W) == (q_vals.size, phi_vals.size):
        return img2d.T, q_vals, phi_vals

    raise ValueError(
        f"Shape mismatch: img={img2d.shape}, len(phi)={phi_vals.size}, len(q)={q_vals.size}. "
        "Expected img=(len(phi),len(q)) or transposed."
    )


def _winsorize(x, p_hi=99.9):
    x = np.asarray(x, dtype=np.float64)
    hi = np.nanpercentile(x, float(p_hi))
    lo = np.nanpercentile(x, 0.0)
    return np.clip(x, lo, hi), float(lo), float(hi)


def _local_median_3x3(img):
    """
    Fast 3x3 local median using shifted stacks (pure numpy, no scipy).
    Edge-handled by padding with edge values.
    """
    a = np.asarray(img, dtype=np.float64)
    p = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    shifts = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifts.append(p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]])
    stack = np.stack(shifts, axis=0)  # (9,H,W)
    return np.median(stack, axis=0)


def _despike_hot_pixels(img, *, z_thresh=12.0, use_log=True):
    """
    Replace extreme outliers using a robust z-score on (optionally) log1p(img).
    Replacement value is local 3x3 median.
    """
    x = np.asarray(img, dtype=np.float64)
    x0 = np.clip(x, 0.0, np.inf)

    y = np.log1p(x0) if use_log else x0
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    if not np.isfinite(mad) or mad == 0:
        return x0, np.zeros_like(x0, dtype=bool)

    # 1.4826 * MAD ~ sigma for normal, good robust scale
    z = (y - med) / (1.4826 * mad)
    hot = z > float(z_thresh)

    if np.any(hot):
        local_med = _local_median_3x3(x0)
        x0 = x0.copy()
        x0[hot] = local_med[hot]

    return x0, hot


def _weighted_quantile(x, w, qs):
    """
    Weighted quantile(s) of x with weights w.
    qs in [0,1] list/array.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    qs = np.asarray(qs, dtype=np.float64).ravel()

    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[m]
    w = w[m]
    if x.size == 0 or np.sum(w) <= 0:
        return np.full(qs.shape, np.nan, dtype=np.float64)

    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]

    return np.interp(qs, cdf, x)


def _weighted_moments_1d(x, w):
    """
    Returns (mu, sigma, skew) for weighted distribution.
    skew = E[(x-mu)^3] / sigma^3
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()

    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[m]
    w = w[m]
    W = np.sum(w)
    if x.size == 0 or W <= 0:
        return np.nan, np.nan, np.nan

    mu = np.sum(w * x) / W
    m2 = np.sum(w * (x - mu) ** 2) / W
    sigma = np.sqrt(max(m2, 0.0))
    if not np.isfinite(sigma) or sigma == 0:
        return float(mu), float(sigma), np.nan

    m3 = np.sum(w * (x - mu) ** 3) / W
    skew = m3 / (sigma ** 3)
    return float(mu), float(sigma), float(skew)


def bragg_peak_shape_metrics_fixed_q_phi(
    scattering_2d,
    *,
    q_vals,
    phi_vals,
    winsor_p_hi=99.9,
    use_log_weights=True,
    despike=True,
    hot_z_thresh=12.0,
):
    """
    Compute fixed-axis (q and phi) statistics for the Bragg peak shape.

    Returns a dict containing:
      - center_q, center_phi (weighted means)
      - sigma_q, sigma_phi  (weighted std dev)
      - skew_q,  skew_phi   (weighted moment skewness)
      - q_profile, phi_profile (marginals)
      - quantile skew (Bowley) in each axis
      - optional hot-pixel mask + thresholds

    Notes
    -----
    - Tails are included (no masking), but hot pixels are optionally despiked.
    - Robustness is controlled by winsor_p_hi and/or use_log_weights.
    - Axis convention enforced: img shape = (len(phi), len(q)).
    """
    img = _ensure_2d(scattering_2d)
    img, q, phi = _match_axes(img, q_vals, phi_vals)

    x = np.asarray(img, dtype=np.float64)
    x = np.clip(x, 0.0, np.inf)

    hot_mask = None
    if despike:
        x, hot_mask = _despike_hot_pixels(x, z_thresh=float(hot_z_thresh), use_log=True)

    # Winsor cap to prevent a few extreme pixels dominating moments
    xw, lo, hi = _winsorize(x, p_hi=float(winsor_p_hi))

    # Weights (optionally log-compressed)
    w = np.log1p(xw) if bool(use_log_weights) else xw
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, np.inf)

    # Marginals: axis0=phi, axis1=q
    Wq = np.sum(w, axis=0)   # (nq,)
    Wphi = np.sum(w, axis=1) # (nphi,)

    # Fixed-axis moments
    mu_q, sig_q, skew_q = _weighted_moments_1d(q, Wq)
    mu_phi, sig_phi, skew_phi = _weighted_moments_1d(phi, Wphi)

    # Quantile (Bowley) skewness for stability check
    q25, q50, q75 = _weighted_quantile(q, Wq, [0.25, 0.50, 0.75])
    p25, p50, p75 = _weighted_quantile(phi, Wphi, [0.25, 0.50, 0.75])

    def _bowley(a25, a50, a75):
        den = (a75 - a25)
        if not np.isfinite(den) or den == 0:
            return np.nan
        return float((a75 + a25 - 2.0 * a50) / den)

    bowley_q = _bowley(q25, q50, q75)
    bowley_phi = _bowley(p25, p50, p75)

    # Effective number of pixels (for “how concentrated are the weights”)
    wflat = w.ravel()
    sw = np.sum(wflat)
    sw2 = np.sum(wflat * wflat)
    neff = (sw * sw / sw2) if (sw2 > 0) else np.nan

    return {
        "center_q": mu_q,
        "center_phi": mu_phi,
        "sigma_q": sig_q,
        "sigma_phi": sig_phi,
        "skew_q": skew_q,
        "skew_phi": skew_phi,
        "bowley_skew_q": bowley_q,
        "bowley_skew_phi": bowley_phi,
        "q_profile": Wq,
        "phi_profile": Wphi,
        "q_vals": q,
        "phi_vals": phi,
        "winsor_lo": lo,
        "winsor_hi": hi,
        "use_log_weights": bool(use_log_weights),
        "despike": bool(despike),
        "hot_pixel_mask": hot_mask,
        "neff": float(neff) if np.isfinite(neff) else np.nan,
    }

def plot_bragg_peak_shape_metrics_overlay_from_maps(
    scattering_2d: np.ndarray,
    *,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    metrics: dict,
    valid_mask: np.ndarray | None = None,
    cmap: str = "magma",
    n_q_contours: int = 7,
    n_phi_contours: int = 7,
):
    """
    Overlay iso-q and iso-phi contours on detector-space scattering_2d,
    plus markers for (q_mean, phi_mean) projected back to pixels.

    This is the detector-space counterpart to the old binned overlay.
    """
    img = np.asarray(scattering_2d)
    if img.ndim == 3:
        img = img[0]
    img = np.asarray(img, dtype=np.float64)

    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)

    if img.shape != q_map.shape or img.shape != phi_map.shape:
        raise ValueError(f"Shape mismatch: img={img.shape}, q_map={q_map.shape}, phi_map={phi_map.shape}")

    if valid_mask is None:
        valid = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)

    if not np.any(valid):
        raise RuntimeError("No valid pixels for overlay.")

    # log-ish display without blowing out
    disp = np.log10(np.clip(img, 0.0, None) + 1e-12)
    img_pos = img[img > 0]

    vmin = np.percentile(img_pos, 1.0)
    vmax = np.percentile(img_pos, 99.9)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.2))
    im = ax.imshow(
        img,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title("Detector-space scattering_2d with q/phi contour overlay")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")

    # Contour levels chosen from valid pixels
    qv = q_map[valid]
    phv = phi_map[valid]

    q_lo, q_hi = np.nanpercentile(qv, [2, 98])
    ph_lo, ph_hi = np.nanpercentile(phv, [2, 98])

    q_levels = np.linspace(q_lo, q_hi, int(n_q_contours))
    ph_levels = np.linspace(ph_lo, ph_hi, int(n_phi_contours))

    # Mask invalid pixels for contouring
    q_for_contour = np.where(valid, q_map, np.nan)
    ph_for_contour = np.where(valid, phi_map, np.nan)

    ax.contour(q_for_contour, levels=q_levels, linewidths=0.7, alpha=0.8)
    ax.contour(ph_for_contour, levels=ph_levels, linewidths=0.7, alpha=0.5)

    # Mark the centre (from metrics when available, else centroid over valid)
    q_mean = float(metrics.get("q_mean"))
    phi_mean = float(metrics.get("phi_mean_rad"))
    if "x_mean_px" in metrics and "y_mean_px" in metrics:
        cx = float(metrics["x_mean_px"])
        cy = float(metrics["y_mean_px"])
    else:
        vv = valid
        w = np.clip(img[vv], 0.0, None)
        iy, ix = np.nonzero(vv)
        sw = float(np.sum(w))
        if sw <= 0:
            raise RuntimeError("No positive weight for centroid.")
        cx = float(np.sum(w * ix) / sw)
        cy = float(np.sum(w * iy) / sw)

    ax.plot([cx], [cy], marker="x", markersize=10, mew=2)
    ax.text(
        cx + 200,
        cy + 50,
        f"Pixel centroid (x̄,ȳ)\nq̄={q_mean:.4f}\nφ̄={phi_mean:.4f} rad",
        fontsize=12,
        color="yellow",
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="log10(Intensity + eps)")
    fig.tight_layout()
    plt.show()

    return {"center_px_rc": (float(cy), float(cx))}

def plot_bragg_peak_shape_metrics_overlay(
    scattering_2d,
    *,
    q_vals,
    phi_vals,
    metrics: dict,
    title: str = "Bragg peak shape metrics (fixed q/phi axes)",
    cmap: str = "magma",
):
    """
    Standard overlay plot:
      - 2D image (log display) with (center_q, center_phi) marker
      - marginal profiles Wq(q) and Wphi(phi)
    """
    img = _ensure_2d(scattering_2d)
    img, q, phi = _match_axes(img, q_vals, phi_vals)

    # Display image as log1p for visibility without thresholding
    disp = np.log1p(np.clip(img, 0.0, np.inf))

    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(
        disp,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=[q[0], q[-1], phi[0], phi[-1]],
    )
    ax0.plot([metrics["center_q"]], [metrics["center_phi"]], marker="x", markersize=10, mew=2)
    ax0.set_xlabel("q")
    ax0.set_ylabel("phi")
    ax0.set_title("log1p(scattering_2d) + center")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(q, metrics["q_profile"], lw=1.8)
    ax1.set_xlabel("q")
    ax1.set_title(
        f"Wq(q)\n"
        f"μ={metrics['center_q']:.6g}, σ={metrics['sigma_q']:.4g}, "
        f"skew={metrics['skew_q']:.3g}"
    )
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(phi, metrics["phi_profile"], lw=1.8)
    ax2.set_xlabel("phi")
    ax2.set_title(
        f"Wφ(φ)\n"
        f"μ={metrics['center_phi']:.6g}, σ={metrics['sigma_phi']:.4g}, "
        f"skew={metrics['skew_phi']:.3g}"
    )
    ax2.grid(True, alpha=0.25)

    fig.suptitle(
        f"{title}\n"
        f"Bowley skew: q={metrics['bowley_skew_q']:.3g}, phi={metrics['bowley_skew_phi']:.3g} | "
        f"Neff={metrics['neff']:.1f}",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Skew-normal fit and brightest-region helpers (for bragg-peak-metrics)
# ---------------------------------------------------------------------------

def _fit_skewnorm_weighted_1d(x, w, *, refine_mle=False, eps=1e-12):
    """
    Fit a skew-normal (Azzalini) to weighted 1D data. Returns loc, scale, shape (a).

    Uses moment matching; optionally refines with weighted MLE.
    Returns dict: loc, scale, shape, skew_fitted, (and on failure: fallback mean/sigma/skew).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    w = np.clip(w, 0.0, None)
    sw = float(np.sum(w))
    if x.size == 0 or sw <= 0 or scipy_stats is None:
        mu = np.nan
        sig = np.nan
        skew = np.nan
        return {"loc": mu, "scale": sig, "shape": 0.0, "skew_fitted": skew, "moment_skew": skew}

    mu = float(np.sum(w * x) / sw)
    m2 = float(np.sum(w * (x - mu) ** 2) / sw)
    sig = float(np.sqrt(max(m2, 0.0)))
    if sig <= 0:
        return {"loc": mu, "scale": 0.0, "shape": 0.0, "skew_fitted": np.nan, "moment_skew": np.nan}
    m3 = float(np.sum(w * (x - mu) ** 3) / sw)
    skew_emp = m3 / (sig ** 3 + eps)
    # Skew-normal skewness is in (-1, 1) approx; clip so we can invert
    skew_emp = float(np.clip(skew_emp, -0.99, 0.99))

    def skewnorm_skew(a):
        if np.isnan(a):
            return np.nan
        try:
            return float(scipy_stats.skewnorm.stats(a, moments="s"))
        except Exception:
            return np.nan

    # Find shape (a) such that skewnorm(a) has skewness = skew_emp
    try:
        if scipy_optimize is not None:
            def obj(a):
                return skewnorm_skew(a) - skew_emp
            if obj(-10) * obj(10) < 0:
                a_fit = float(scipy_optimize.brentq(obj, -10, 10, xtol=1e-6))
            else:
                a_fit = 0.0
        else:
            a_fit = 0.0
    except Exception:
        a_fit = 0.0

    # Location and scale from mean and std of skewnorm(a, 0, 1)
    try:
        m1, m2_sn = float(scipy_stats.skewnorm.stats(a_fit, moments="m")), float(
            scipy_stats.skewnorm.stats(a_fit, moments="v")
        )
        scale = sig / (np.sqrt(m2_sn) + eps)
        loc = mu - scale * m1
    except Exception:
        loc, scale = mu, sig

    if refine_mle and scipy_optimize is not None and np.isfinite(loc) and scale > 0:
        def nll(params):
            a, loc_p, scale_p = params
            if scale_p <= 0:
                return 1e20
            try:
                logp = scipy_stats.skewnorm.logpdf(x, a, loc=loc_p, scale=scale_p)
                return -float(np.sum(w * np.where(np.isfinite(logp), logp, -1e10)))
            except Exception:
                return 1e20
        try:
            res = scipy_optimize.minimize(
                nll,
                [a_fit, loc, scale],
                method="L-BFGS-B",
                bounds=[(-20, 20), (None, None), (eps, None)],
                options={"maxiter": 200},
            )
            if res.success:
                a_fit, loc, scale = res.x
        except Exception:
            pass

    try:
        skew_fitted = float(scipy_stats.skewnorm.stats(a_fit, moments="s"))
    except Exception:
        skew_fitted = skew_emp

    return {
        "loc": float(loc),
        "scale": float(scale),
        "shape": float(a_fit),
        "skew_fitted": skew_fitted,
        "moment_skew": skew_emp,
    }


def _brightest_region_mask(
    img: np.ndarray,
    valid: np.ndarray,
    *,
    method: str = "roi_around_argmax",
    roi_half_size: int = 250,
    brightest_percentile: float = 90.0,
    brightest_connected: bool = True,
) -> np.ndarray:
    """
    Return a boolean mask of the "brightest region" for Bragg peak analysis.

    method:
      - "roi_around_argmax": fixed (2*roi_half_size+1) box around argmax pixel (current behaviour).
      - "percentile": pixels with intensity >= brightest_percentile (of valid pixels).
      - "connected": percentile threshold then keep only the connected component containing argmax.
    """
    img = np.asarray(img, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(img)
    if not np.any(valid):
        return np.zeros_like(valid, dtype=bool)

    if method == "roi_around_argmax":
        iy0, ix0 = np.unravel_index(np.nanargmax(np.where(valid, img, np.nan)), img.shape)
        roi = np.zeros_like(valid, dtype=bool)
        roi[
            max(0, iy0 - roi_half_size) : min(img.shape[0], iy0 + roi_half_size + 1),
            max(0, ix0 - roi_half_size) : min(img.shape[1], ix0 + roi_half_size + 1),
        ] = True
        return roi & valid

    if method == "percentile":
        p = float(np.clip(brightest_percentile, 0.0, 100.0))
        thr = np.nanpercentile(img[valid], p)
        bright = valid & (img >= thr)
        if not brightest_connected or ndi_label is None:
            return bright
        # Fall through to connected: take component containing argmax
        method = "connected"

    if method == "connected":
        p = float(np.clip(brightest_percentile, 0.0, 100.0))
        thr = np.nanpercentile(img[valid], p)
        bright = valid & (img >= thr)
        if not np.any(bright) or ndi_label is None:
            return bright
        labeled, ncomp = ndi_label(bright)
        iy0, ix0 = np.unravel_index(np.nanargmax(np.where(valid, img, np.nan)), img.shape)
        seed_label = labeled[iy0, ix0]
        if seed_label == 0:
            return bright
        return labeled == seed_label

    return np.zeros_like(valid, dtype=bool)


def bragg_peak_shape_metrics_fixed_q_phi_from_maps(
    scattering_2d: np.ndarray,
    *,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    hot_z_thresh: float = 12.0,
    eps: float = 1e-12,
    # Brightest-region options
    brightest_region_method: str = "roi_around_argmax",
    roi_half_size: int = 250,
    brightest_percentile: float = 90.0,
    brightest_connected: bool = True,
    # Skew-normal fit options
    use_skewnorm_fit: bool = True,
    refine_skewnorm_mle: bool = False,
) -> dict:
    """
    Compute Bragg-peak shape metrics with q and phi defined per DETECTOR PIXEL.

    Inputs
    ------
    scattering_2d : (H,W) or (1,H,W)
        Detector-space average image (NOT 120x30 binned).
    q_map, phi_map : (H,W)
        Per-pixel q and phi maps (e.g. from inferred_qphi_maps.npz).
    valid_mask : (H,W) bool or None
        Optional mask of valid pixels. If None, inferred from finiteness.
    hot_z_thresh : float
        Suppress extreme hot pixels using robust z-score on intensity.
    brightest_region_method : str
        "roi_around_argmax" | "percentile" | "connected". How to define the peak region.
    roi_half_size : int
        Half-size of the box (pixels) when method is roi_around_argmax.
    brightest_percentile : float
        Percentile threshold (0–100) for percentile/connected methods (e.g. 90 = top 10%).
    brightest_connected : bool
        If True and method is "percentile", use only the connected component containing argmax.
    use_skewnorm_fit : bool
        If True, fit skew-normal in x, y, q, phi and report centre/scale/shape from fit.
    refine_skewnorm_mle : bool
        If True and use_skewnorm_fit, refine the fit with weighted MLE.

    Returns
    -------
    dict of scalar metrics (centres, sigmas, skewness; moment-based and optionally skewnorm-fit).
    """
    img = np.asarray(scattering_2d)
    if img.ndim == 3:
        if img.shape[0] != 1:
            raise ValueError(f"Expected scattering_2d with leading dim 1, got {img.shape}")
        img = img[0]
    img = np.asarray(img, dtype=np.float64)

    H, W = img.shape
    iy_max, ix_max = np.unravel_index(np.nanargmax(img), img.shape)
    print("argmax (ix, iy):", ix_max, iy_max)
    print("argmax display-y if origin='lower':", (H - 1 - iy_max))
    print("img max:", img[iy_max, ix_max])

    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)

    if img.shape != q_map.shape or img.shape != phi_map.shape:
        raise ValueError(
            f"Shape mismatch: img={img.shape}, q_map={q_map.shape}, phi_map={phi_map.shape}"
        )

    if valid_mask is None:
        valid = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)

    # ---- hot pixel suppression (robust z-score on intensity)
    # This is a minimal, conservative rejection of extreme spikes.
    vv = valid
    if np.any(vv):
        med = np.nanmedian(img[vv])
        mad = np.nanmedian(np.abs(img[vv] - med))
        sigma_rob = 1.4826 * mad + eps
        z = (img - med) / sigma_rob
        vv = vv & (z < float(hot_z_thresh))

    if not np.any(vv):
        raise RuntimeError("No valid pixels after masking / hot-pixel suppression.")

    # Brightest-region mask (replaces fixed box)
    bright_mask = _brightest_region_mask(
        img,
        vv,
        method=brightest_region_method,
        roi_half_size=roi_half_size,
        brightest_percentile=brightest_percentile,
        brightest_connected=brightest_connected,
    )
    vv_used = vv & bright_mask
    iy0, ix0 = np.unravel_index(np.nanargmax(img), img.shape)

    w = img[vv_used].copy()

    # Weights must be non-negative for moment interpretation
    # (if you have negative values from processing, clip them softly)
    w = np.clip(w, 0.0, None)

    iy, ix = np.nonzero(vv_used)

    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= eps:
        raise RuntimeError("Sum of weights is zero or non-finite, cannot compute moments.")

    qv = q_map[vv_used]
    ph = phi_map[vv_used]

    # ---- weighted mean in q
    q_mean = float(np.sum(w * qv) / sw)

    # ---- weighted circular mean in phi
    c = float(np.sum(w * np.cos(ph)) / sw)
    s = float(np.sum(w * np.sin(ph)) / sw)
    phi_mean = float(np.arctan2(s, c))

    # unwrap phi about mean to compute *directional* moments
    dphi = ph - phi_mean
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]

    dq = qv - q_mean

    # ---- weighted central moments and skewness
    mu2_q = float(np.sum(w * dq * dq) / sw)
    mu3_q = float(np.sum(w * dq * dq * dq) / sw)

    mu2_phi = float(np.sum(w * dphi * dphi) / sw)
    mu3_phi = float(np.sum(w * dphi * dphi * dphi) / sw)

    sigma_q = float(np.sqrt(max(mu2_q, 0.0)))
    sigma_phi = float(np.sqrt(max(mu2_phi, 0.0)))

    skew_q = float(mu3_q / (mu2_q ** 1.5 + eps))
    skew_phi = float(mu3_phi / (mu2_phi ** 1.5 + eps))

    # Optional extra diagnostics that are often useful
    peak_adu = float(np.nanmax(img[vv_used]))
    n_pix = int(np.sum(vv_used))
    frac_kept = float(n_pix / np.sum(valid)) if np.sum(valid) > 0 else float("nan")

    # ----------------------------
    # Pixel-space (x, y) metrics
    # ----------------------------
    # iy, ix = np.nonzero(valid_mask)

    w_sum = np.sum(w)

    # centroids in pixel coordinates
    x_mean = np.sum(w * ix) / w_sum
    y_mean = np.sum(w * iy) / w_sum

    # second moments
    sigma_x = np.sqrt(np.sum(w * (ix - x_mean) ** 2) / w_sum)
    sigma_y = np.sqrt(np.sum(w * (iy - y_mean) ** 2) / w_sum)

    # skewness (moment-based)
    skew_x = np.sum(w * (ix - x_mean) ** 3) / (w_sum * (sigma_x ** 3 + eps))
    skew_y = np.sum(w * (iy - y_mean) ** 3) / (w_sum * (sigma_y ** 3 + eps))

    # ----------------------------
    # Skew-normal fit (optional)
    # ----------------------------
    out_x_mean = float(x_mean)
    out_y_mean = float(y_mean)
    out_sigma_x = float(sigma_x)
    out_sigma_y = float(sigma_y)
    out_skew_x = float(skew_x)
    out_skew_y = float(skew_y)
    out_q_mean = q_mean
    out_phi_mean = phi_mean
    out_sigma_q = sigma_q
    out_sigma_phi = sigma_phi
    out_skew_q = skew_q
    out_skew_phi = skew_phi
    shape_x = shape_y = shape_q = shape_phi = np.nan
    skew_x_fitted = skew_y_fitted = skew_q_fitted = skew_phi_fitted = np.nan

    if use_skewnorm_fit and scipy_stats is not None:
        fit_x = _fit_skewnorm_weighted_1d(ix, w, refine_mle=refine_skewnorm_mle, eps=eps)
        fit_y = _fit_skewnorm_weighted_1d(iy, w, refine_mle=refine_skewnorm_mle, eps=eps)
        fit_q = _fit_skewnorm_weighted_1d(qv, w, refine_mle=refine_skewnorm_mle, eps=eps)
        fit_phi = _fit_skewnorm_weighted_1d(dphi, w, refine_mle=refine_skewnorm_mle, eps=eps)
        out_x_mean = fit_x["loc"]
        out_y_mean = fit_y["loc"]
        out_sigma_x = fit_x["scale"]
        out_sigma_y = fit_y["scale"]
        out_skew_x = fit_x["skew_fitted"]
        out_skew_y = fit_y["skew_fitted"]
        shape_x = fit_x["shape"]
        shape_y = fit_y["shape"]
        skew_x_fitted = fit_x["skew_fitted"]
        skew_y_fitted = fit_y["skew_fitted"]
        out_q_mean = fit_q["loc"]
        out_sigma_q = fit_q["scale"]
        out_skew_q = fit_q["skew_fitted"]
        shape_q = fit_q["shape"]
        skew_q_fitted = fit_q["skew_fitted"]
        # phi: fit was in unwrapped space; centre in original = phi_mean + loc
        out_phi_mean = float((phi_mean + fit_phi["loc"] + np.pi) % (2.0 * np.pi) - np.pi)
        out_sigma_phi = fit_phi["scale"]
        out_skew_phi = fit_phi["skew_fitted"]
        shape_phi = fit_phi["shape"]
        skew_phi_fitted = fit_phi["skew_fitted"]

    H, W = img.shape
    print("argmax array (x,y):", ix0, iy0, " display-y:", (H - 1) - iy0)
    print("centroid array (x,y):", out_x_mean, out_y_mean, " display-y:", (H - 1) - out_y_mean)

    result = {
        "x_mean_px": out_x_mean,
        "y_mean_px": out_y_mean,
        "sigma_x_px": out_sigma_x,
        "sigma_y_px": out_sigma_y,
        "skew_x": out_skew_x,
        "skew_y": out_skew_y,
        "q_mean": out_q_mean,
        "phi_mean_rad": out_phi_mean,
        "sigma_q": out_sigma_q,
        "sigma_phi_rad": out_sigma_phi,
        "skew_q": out_skew_q,
        "skew_phi": out_skew_phi,
        "n_pix_used": n_pix,
        "frac_valid_used": frac_kept,
        "peak_intensity": peak_adu,
        "hot_z_thresh": float(hot_z_thresh),
        "brightest_region_method": brightest_region_method,
        "use_skewnorm_fit": use_skewnorm_fit,
    }
    if use_skewnorm_fit and scipy_stats is not None:
        result["shape_x"] = float(shape_x)
        result["shape_y"] = float(shape_y)
        result["shape_q"] = float(shape_q)
        result["shape_phi"] = float(shape_phi)
        result["skew_x_fitted"] = float(skew_x_fitted)
        result["skew_y_fitted"] = float(skew_y_fitted)
        result["skew_q_fitted"] = float(skew_q_fitted)
        result["skew_phi_fitted"] = float(skew_phi_fitted)
    return result


# ---------------------------------------------------------------------------
# q/φ maps from static qmap
# ---------------------------------------------------------------------------

def infer_q_phi_maps_from_static_qmap(
    f,
    *,
    q_key: str = "xpcs/qmap/static_v_list_dim0",
    phi_key: str = "xpcs/qmap/static_v_list_dim1",
    roi_map_key: str = "xpcs/qmap/static_roi_map",
    index_map_key: str = "xpcs/qmap/static_index_mapping",
    num_pts_key: str = "xpcs/qmap/static_num_pts",
    invalid_roi_value: int = 0,
):
    """
    Build per-pixel q_map / phi_map (and pseudo-Qx/Qy) from the static qmap products.

    Requires:
      - static_roi_map: per-pixel ROI id
      - static_index_mapping: length Nq*Nphi, maps (iq,iphi) bin -> ROI id
      - static_v_list_dim0: q bin centers (Nq)
      - static_v_list_dim1: phi bin centers (Nphi)

    Returns
    -------
    out : dict with keys
      q_map, phi_map, Qx_map, Qy_map, valid_mask, roi_map, q_vals, phi_vals

    Notes
    -----
    - phi may be in degrees or radians; we infer by range and convert to radians for cos/sin.
    - invalid_roi_value (often 0) is treated as background/invalid.
    """
    q_vals = np.asarray(f[q_key][...], dtype=np.float64)          # (Nq,)
    phi_vals = np.asarray(f[phi_key][...], dtype=np.float64)      # (Nphi,)
    roi_map = np.asarray(f[roi_map_key][...], dtype=np.int64)     # (ny,nx)

    # index_mapping: length Nq*Nphi, values are ROI ids (uint16)
    idx_to_roi = np.asarray(f[index_map_key][...], dtype=np.int64)

    # sanity check Nq,Nphi from file (optional but helpful)
    if num_pts_key in f:
        npts = np.asarray(f[num_pts_key][...], dtype=np.int64).ravel()
        if npts.size == 2:
            Nq_file, Nphi_file = int(npts[0]), int(npts[1])
            if Nq_file != q_vals.size or Nphi_file != phi_vals.size:
                raise ValueError(
                    f"static_num_pts says (Nq,Nphi)=({Nq_file},{Nphi_file}) "
                    f"but v_list sizes are (Nq,Nphi)=({q_vals.size},{phi_vals.size})"
                )

    Nq = int(q_vals.size)
    Nphi = int(phi_vals.size)
    if idx_to_roi.size != Nq * Nphi:
        raise ValueError(
            f"static_index_mapping has length {idx_to_roi.size}, expected Nq*Nphi={Nq*Nphi}"
        )

    # infer phi units -> radians
    # (if values look like degrees, convert)
    phi_max = float(np.nanmax(phi_vals))
    phi_min = float(np.nanmin(phi_vals))
    # crude but reliable for your typical bin-centers
    if (phi_max - phi_min) > (2.0 * np.pi + 0.5):
        phi_rad = np.deg2rad(phi_vals)
    else:
        phi_rad = phi_vals.copy()

    # Build ROI->(q,phi) lookup.
    # idx = iq*Nphi + iphi  (iphi changes fastest, matches your "repeats every 30")
    # idx_to_roi[idx] gives ROI id
    roi_max = int(np.max(roi_map))
    roi_to_q = np.full((roi_max + 1,), np.nan, dtype=np.float64)
    roi_to_phi = np.full((roi_max + 1,), np.nan, dtype=np.float64)

    for iq in range(Nq):
        for iphi in range(Nphi):
            lin = iq * Nphi + iphi
            roi_id = int(idx_to_roi[lin])
            if roi_id <= 0 or roi_id > roi_max:
                continue
            # assign (q,phi) for that ROI id
            # if duplicates exist, they should match; if not, last wins
            roi_to_q[roi_id] = float(q_vals[iq])
            roi_to_phi[roi_id] = float(phi_rad[iphi])

    # Now map per-pixel
    valid_mask = (roi_map != int(invalid_roi_value)) & (roi_map >= 0) & (roi_map <= roi_max)

    q_map = np.full_like(roi_map, np.nan, dtype=np.float64)
    phi_map = np.full_like(roi_map, np.nan, dtype=np.float64)

    q_map[valid_mask] = roi_to_q[roi_map[valid_mask]]
    phi_map[valid_mask] = roi_to_phi[roi_map[valid_mask]]

    # pseudo-components
    Qx_map = np.full_like(q_map, np.nan, dtype=np.float64)
    Qy_map = np.full_like(q_map, np.nan, dtype=np.float64)

    vv = valid_mask & np.isfinite(q_map) & np.isfinite(phi_map)
    Qx_map[vv] = q_map[vv] * np.cos(phi_map[vv])
    Qy_map[vv] = q_map[vv] * np.sin(phi_map[vv])

    return {
        "q_map": q_map,
        "phi_map": phi_map,
        "Qx_map": Qx_map,
        "Qy_map": Qy_map,
        "valid_mask": vv,
        "roi_map": roi_map,
        "q_vals": q_vals,
        "phi_vals": phi_vals,
    }


def save_inferred_qphi_maps_npz(
    results_hdf_path: str | Path,
    out_npz_path: str | Path,
):
    """
    Convenience wrapper: open results .hdf, infer maps, save to .npz.
    """
    results_hdf_path = Path(results_hdf_path)
    out_npz_path = Path(out_npz_path)

    with h5py.File(results_hdf_path, "r") as f:
        # your results file stores under /xpcs/...
        maps = infer_q_phi_maps_from_static_qmap(f, q_key="xpcs/qmap/static_v_list_dim0",
                                                 phi_key="xpcs/qmap/static_v_list_dim1",
                                                 roi_map_key="xpcs/qmap/static_roi_map",
                                                 index_map_key="xpcs/qmap/static_index_mapping",
                                                 num_pts_key="xpcs/qmap/static_num_pts")

    np.savez_compressed(
        out_npz_path,
        q_map=maps["q_map"],
        phi_map=maps["phi_map"],
        Qx_map=maps["Qx_map"],
        Qy_map=maps["Qy_map"],
        valid_mask=maps["valid_mask"],
        roi_map=maps["roi_map"],
        q_vals=maps["q_vals"],
        phi_vals=maps["phi_vals"],
    )
    print(f"Saved inferred maps: {out_npz_path}")
    return out_npz_path


# ---------------------------------------------------------------------------
# Config defaults and path resolution (used by CLI and execution functions)
# ---------------------------------------------------------------------------

DEFAULT_FILE_ID = "A073"
DEFAULT_BASE_DIR = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")
# Overrides for specific file_ids when not using base_dir glob
FILE_ID_OVERRIDES = {
    "A013": Path("/Users/emilioescauriza/Desktop/A013_IPA_NBH_1_att0100_079K_001_results.hdf"),
    "A073": Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_IPA_NBH_1_att0100_260K_001_results.hdf"),
}

# Set by CLI or for backward compatibility when run without CLI
filename = None
h5_file = None


def resolve_results_path(file_id: str, base_dir: Path | None = None) -> Path:
    """Return path to results HDF for the given file_id. Uses overrides or base_dir glob."""
    if file_id in FILE_ID_OVERRIDES:
        return FILE_ID_OVERRIDES[file_id]
    base = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    matches = sorted(base.glob(f"{file_id}_*_results.hdf"))
    if not matches:
        raise FileNotFoundError(f"No results HDF found for file_id={file_id!r} in {base}")
    return matches[0]


# ---------------------------------------------------------------------------
# Brightest-region centre + overlay; skew-normal from centre (lineouts x,y; region q,φ)
# ---------------------------------------------------------------------------

def get_brightest_region_centre(
    scattering_2d: np.ndarray,
    *,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    brightest_region_method: str = "connected",
    roi_half_size: int = 250,
    brightest_percentile: float = 90.0,
    brightest_connected: bool = True,
):
    """
    Compute centre of brightest region (intensity-weighted centroid) and weighted-mean q, φ.
    Returns dict with cx, cy, q, phi_rad, region (bool mask), n_region, intensity_max_region.
    """
    img = np.asarray(scattering_2d, dtype=np.float64)
    if img.ndim == 3:
        img = img[0]
    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)
    if img.shape != q_map.shape or img.shape != phi_map.shape:
        raise ValueError(f"Shape mismatch: img={img.shape}, q_map={q_map.shape}, phi_map={phi_map.shape}")
    if valid_mask is None:
        valid = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    if not np.any(valid):
        raise RuntimeError("No valid pixels.")
    region = _brightest_region_mask(
        img, valid,
        method=brightest_region_method,
        roi_half_size=roi_half_size,
        brightest_percentile=brightest_percentile,
        brightest_connected=brightest_connected,
    )
    region = region & valid
    if not np.any(region):
        raise RuntimeError("Brightest region is empty.")
    w = np.clip(img[region], 0.0, None)
    iy, ix = np.nonzero(region)
    sw = float(np.sum(w))
    if sw <= 0:
        raise RuntimeError("Sum of weights is zero in brightest region.")
    cx = float(np.sum(w * ix) / sw)
    cy = float(np.sum(w * iy) / sw)
    q_centre = float(np.sum(w * q_map[region]) / sw)
    c = float(np.sum(w * np.cos(phi_map[region])) / sw)
    s = float(np.sum(w * np.sin(phi_map[region])) / sw)
    phi_centre = float(np.arctan2(s, c))
    return {
        "cx": cx, "cy": cy, "q": q_centre, "phi_rad": phi_centre,
        "region": region, "n_region": int(np.sum(region)),
        "intensity_max_region": float(np.nanmax(img[region])),
    }


def bragg_peak_skewnorm_from_centre(
    scattering_2d: np.ndarray,
    *,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    cx: float,
    cy: float,
    region_mask: np.ndarray | None = None,
    region_half_size: int = 200,
    valid_mask: np.ndarray | None = None,
    refine_mle: bool = False,
    lineout_half_width: int = 80,
    eps: float = 1e-12,
) -> dict:
    """
    Given centre (cx, cy) in pixels, fit skew-normal in x and y via lineouts,
    and in q and φ via a region around the centre.

    - x: lineout row at round(cy), fit (ix, intensity) in a window of ±lineout_half_width around the peak → loc_x, scale_x, shape_x.
    - y: lineout column at round(cx), fit (iy, intensity) in a window around the peak.
    - q, φ: use region_mask if provided, else box around (cx, cy); fit (q, w) and (φ unwrapped, w).
    lineout_half_width: only use pixels within this many pixels of the lineout peak when fitting x and y (reduces tail pull, narrower fit).
    """
    img = np.asarray(scattering_2d, dtype=np.float64)
    if img.ndim == 3:
        img = img[0]
    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)
    H, W = img.shape
    if valid_mask is None:
        valid = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)

    out = {"cx": cx, "cy": cy}

    # Lineout x: row at cy; fit only in a window around the peak so tails don't inflate scale
    row = int(np.round(np.clip(cy, 0, H - 1)))
    ix = np.arange(W, dtype=np.float64)
    w_x = np.clip(img[row, :], 0.0, None)
    if np.sum(w_x) > eps:
        mu_x = float(np.sum(w_x * ix) / np.sum(w_x))
        i_lo = max(0, int(mu_x - lineout_half_width))
        i_hi = min(W, int(mu_x + lineout_half_width) + 1)
        ix_win = ix[i_lo:i_hi]
        w_x_win = w_x[i_lo:i_hi]
        if np.sum(w_x_win) > eps:
            fit_x = _fit_skewnorm_weighted_1d(ix_win, w_x_win, refine_mle=True, eps=eps)
            out["x_loc"] = fit_x["loc"]; out["x_scale"] = fit_x["scale"]; out["x_shape"] = fit_x["shape"]; out["x_skew"] = fit_x["skew_fitted"]
        else:
            fit_x = _fit_skewnorm_weighted_1d(ix, w_x, refine_mle=refine_mle, eps=eps)
            out["x_loc"] = fit_x["loc"]; out["x_scale"] = fit_x["scale"]; out["x_shape"] = fit_x["shape"]; out["x_skew"] = fit_x["skew_fitted"]
    else:
        out["x_loc"] = out["x_scale"] = out["x_shape"] = out["x_skew"] = np.nan

    # Lineout y: column at cx; same windowing
    col = int(np.round(np.clip(cx, 0, W - 1)))
    iy = np.arange(H, dtype=np.float64)
    w_y = np.clip(img[:, col], 0.0, None)
    if np.sum(w_y) > eps:
        mu_y = float(np.sum(w_y * iy) / np.sum(w_y))
        j_lo = max(0, int(mu_y - lineout_half_width))
        j_hi = min(H, int(mu_y + lineout_half_width) + 1)
        iy_win = iy[j_lo:j_hi]
        w_y_win = w_y[j_lo:j_hi]
        if np.sum(w_y_win) > eps:
            fit_y = _fit_skewnorm_weighted_1d(iy_win, w_y_win, refine_mle=True, eps=eps)
            out["y_loc"] = fit_y["loc"]; out["y_scale"] = fit_y["scale"]; out["y_shape"] = fit_y["shape"]; out["y_skew"] = fit_y["skew_fitted"]
        else:
            fit_y = _fit_skewnorm_weighted_1d(iy, w_y, refine_mle=refine_mle, eps=eps)
            out["y_loc"] = fit_y["loc"]; out["y_scale"] = fit_y["scale"]; out["y_shape"] = fit_y["shape"]; out["y_skew"] = fit_y["skew_fitted"]
    else:
        out["y_loc"] = out["y_scale"] = out["y_shape"] = out["y_skew"] = np.nan

    # Region for q, φ
    if region_mask is not None:
        reg = np.asarray(region_mask, dtype=bool) & valid
    else:
        y0, x0 = int(np.round(cy)), int(np.round(cx))
        reg = np.zeros_like(valid, dtype=bool)
        reg[max(0, y0 - region_half_size):min(H, y0 + region_half_size + 1),
            max(0, x0 - region_half_size):min(W, x0 + region_half_size + 1)] = True
        reg = reg & valid
    if not np.any(reg):
        out["q_loc"] = out["q_scale"] = out["q_shape"] = out["q_skew"] = np.nan
        out["phi_loc"] = out["phi_scale"] = out["phi_shape"] = out["phi_skew"] = np.nan
        return out
    w = np.clip(img[reg], 0.0, None)
    q_vals = q_map[reg]
    phi_vals = phi_map[reg]
    phi_centre = float(np.arctan2(np.sum(w * np.sin(phi_vals)), np.sum(w * np.cos(phi_vals))))
    dphi = (phi_vals - phi_centre + np.pi) % (2.0 * np.pi) - np.pi
    if np.sum(w) > eps:
        fit_q = _fit_skewnorm_weighted_1d(q_vals, w, refine_mle=refine_mle, eps=eps)
        out["q_loc"] = fit_q["loc"]; out["q_scale"] = fit_q["scale"]; out["q_shape"] = fit_q["shape"]; out["q_skew"] = fit_q["skew_fitted"]
        fit_phi = _fit_skewnorm_weighted_1d(dphi, w, refine_mle=refine_mle, eps=eps)
        out["phi_loc_unwrapped"] = fit_phi["loc"]
        out["phi_centre_rad"] = phi_centre
        out["phi_loc"] = (phi_centre + fit_phi["loc"] + np.pi) % (2.0 * np.pi) - np.pi
        out["phi_scale"] = fit_phi["scale"]; out["phi_shape"] = fit_phi["shape"]; out["phi_skew"] = fit_phi["skew_fitted"]
    else:
        out["q_loc"] = out["q_scale"] = out["q_shape"] = out["q_skew"] = np.nan
        out["phi_loc"] = out["phi_scale"] = out["phi_shape"] = out["phi_skew"] = np.nan
    return out


def plot_brightest_pixel_qphi_overlay(
    scattering_2d: np.ndarray,
    *,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    cmap: str = "magma",
    brightest_region_method: str = "connected",
    roi_half_size: int = 250,
    brightest_percentile: float = 90.0,
    brightest_connected: bool = True,
    centre_info: dict | None = None,
    skewnorm_result: dict | None = None,
):
    """
    Find the centre of the brightest region (or use centre_info if provided), draw a cross and label with q, φ.
    If skewnorm_result is provided, add skew annotations and x/y lineout panels with skew-normal fit overlaid.
    """
    img = np.asarray(scattering_2d, dtype=np.float64)
    if img.ndim == 3:
        img = img[0]
    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)
    H, W = img.shape
    if img.shape != q_map.shape or img.shape != phi_map.shape:
        raise ValueError(f"Shape mismatch: img={img.shape}, q_map={q_map.shape}, phi_map={phi_map.shape}")
    if valid_mask is None:
        valid = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    if not np.any(valid):
        raise RuntimeError("No valid pixels.")
    if centre_info is None:
        info = get_brightest_region_centre(
            scattering_2d, q_map=q_map, phi_map=phi_map, valid_mask=valid_mask,
            brightest_region_method=brightest_region_method, roi_half_size=roi_half_size,
            brightest_percentile=brightest_percentile, brightest_connected=brightest_connected,
        )
    else:
        info = centre_info
    cx, cy = info["cx"], info["cy"]
    q_at, phi_at = info["q"], info["phi_rad"]
    vmin = np.nanpercentile(img[valid], 1)
    vmax = np.nanpercentile(img[valid], 99.999)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = np.nanmax(img[valid])
    norm = LogNorm(vmin=vmin, vmax=vmax)

    if skewnorm_result is not None:
        fig = plt.figure(figsize=(7.4, 8.0))
        gs = GridSpec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.25)
        ax = fig.add_subplot(gs[0, :])
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.2))

    im = ax.imshow(img, origin="upper", cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    ax.set_title("Centre of brightest region (Bragg peak)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")
    ax.plot([cx], [cy], marker="x", markersize=14, mew=2.5, color="yellow")
    ax.text(cx + 15, cy - 15, f"q = {q_at:.5f}\nφ = {phi_at:.5f} rad", fontsize=11, color="yellow", verticalalignment="top")
    if skewnorm_result is not None:
        sk = skewnorm_result
        txt = (
            f"skew_x = {sk.get('x_skew', np.nan):.3f}  skew_y = {sk.get('y_skew', np.nan):.3f}\n"
            f"skew_q = {sk.get('q_skew', np.nan):.3f}  skew_φ = {sk.get('phi_skew', np.nan):.3f}"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9, color="white", verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Intensity")

    if skewnorm_result is not None and scipy_stats is not None:
        sk = skewnorm_result
        row = int(np.round(np.clip(cy, 0, H - 1)))
        col = int(np.round(np.clip(cx, 0, W - 1)))
        ix = np.arange(W, dtype=np.float64)
        iy = np.arange(H, dtype=np.float64)
        line_x = np.clip(img[row, :], 0.0, None)
        line_y = np.clip(img[:, col], 0.0, None)
        ax_x = fig.add_subplot(gs[1, 0])
        ax_y = fig.add_subplot(gs[1, 1])
        ax_x.plot(ix, line_x, "C0-", lw=0.8, label="lineout", alpha=0.8)
        ax_y.plot(iy, line_y, "C0-", lw=0.8, label="lineout", alpha=0.8)
        if np.isfinite(sk.get("x_loc")) and np.isfinite(sk.get("x_scale")) and sk.get("x_scale", 0) > 0:
            pdf_x = scipy_stats.skewnorm.pdf(ix, sk["x_shape"], sk["x_loc"], sk["x_scale"])
            scale_x = np.nanmax(line_x) / (np.nanmax(pdf_x) + 1e-12)
            ax_x.plot(ix, pdf_x * scale_x, "C1-", lw=1.2, label="skew-normal fit")
        ax_x.set_xlabel("x (pixel)"); ax_x.set_ylabel("Intensity"); ax_x.set_title("X lineout"); ax_x.legend(loc="upper right", fontsize=8); ax_x.grid(True, alpha=0.3)
        if np.isfinite(sk.get("y_loc")) and np.isfinite(sk.get("y_scale")) and sk.get("y_scale", 0) > 0:
            pdf_y = scipy_stats.skewnorm.pdf(iy, sk["y_shape"], sk["y_loc"], sk["y_scale"])
            scale_y = np.nanmax(line_y) / (np.nanmax(pdf_y) + 1e-12)
            ax_y.plot(iy, pdf_y * scale_y, "C1-", lw=1.2, label="skew-normal fit")
        ax_y.set_xlabel("y (pixel)"); ax_y.set_ylabel("Intensity"); ax_y.set_title("Y lineout"); ax_y.legend(loc="upper right", fontsize=8); ax_y.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    return {**info, "intensity_max_region": info["intensity_max_region"]}


def exec_bragg_peak_brightest(
    *,
    brightest_region_method: str = "connected",
    roi_half_size: int = 250,
    brightest_percentile: float = 90.0,
    brightest_connected: bool = True,
):
    """Load scattering_2d and q/φ maps, find centre of brightest region, show cross + q, φ label."""
    with h5py.File(filename, "r") as f:
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
    npz_path = filename.parent / (filename.stem.replace("_results", "") + "_inferred_qphi_maps.npz")
    if not npz_path.is_file():
        npz_path = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")
    d = np.load(npz_path, allow_pickle=False)
    valid = d.get("valid_mask", None)
    centre = get_brightest_region_centre(
        scattering_2d, q_map=d["q_map"], phi_map=d["phi_map"], valid_mask=valid,
        brightest_region_method=brightest_region_method,
        roi_half_size=roi_half_size,
        brightest_percentile=brightest_percentile,
        brightest_connected=brightest_connected,
    )
    skew = bragg_peak_skewnorm_from_centre(
        scattering_2d, q_map=d["q_map"], phi_map=d["phi_map"],
        cx=centre["cx"], cy=centre["cy"], region_mask=centre["region"], valid_mask=valid,
    )
    info = plot_brightest_pixel_qphi_overlay(
        scattering_2d, q_map=d["q_map"], phi_map=d["phi_map"],
        valid_mask=valid, cmap="magma",
        centre_info=centre,
        skewnorm_result=skew,
    )
    print("Centre of brightest region: cx=%.2f, cy=%.2f  q=%.5f  φ=%.5f rad  n_region=%d" % (
        info["cx"], info["cy"], info["q"], info["phi_rad"], info["n_region"]))
    print("Skew-normal (x, lineout): loc=%.2f scale=%.2f shape=%.3f skew=%.3f" % (skew["x_loc"], skew["x_scale"], skew["x_shape"], skew["x_skew"]))
    print("Skew-normal (y, lineout): loc=%.2f scale=%.2f shape=%.3f skew=%.3f" % (skew["y_loc"], skew["y_scale"], skew["y_shape"], skew["y_skew"]))
    print("Skew-normal (q, region):  loc=%.5f scale=%.5f shape=%.3f skew=%.3f" % (skew["q_loc"], skew["q_scale"], skew["q_shape"], skew["q_skew"]))
    print("Skew-normal (φ, region):  loc=%.5f rad scale=%.5f shape=%.3f skew=%.3f" % (skew["phi_loc"], skew["phi_scale"], skew["phi_shape"], skew["phi_skew"]))
    return {**info, "skewnorm": skew}


def exec_bragg_peak_skewnorm(
    *,
    brightest_region_method: str = "connected",
    roi_half_size: int = 250,
    brightest_percentile: float = 90.0,
    brightest_connected: bool = True,
):
    """Centre of brightest region + skew-normal fits in x, y (lineouts) and q, φ (region). Show overlay and print params."""
    with h5py.File(filename, "r") as f:
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
    npz_path = filename.parent / (filename.stem.replace("_results", "") + "_inferred_qphi_maps.npz")
    if not npz_path.is_file():
        npz_path = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")
    d = np.load(npz_path, allow_pickle=False)
    valid = d.get("valid_mask", None)
    centre = get_brightest_region_centre(
        scattering_2d, q_map=d["q_map"], phi_map=d["phi_map"], valid_mask=valid,
        brightest_region_method=brightest_region_method,
        roi_half_size=roi_half_size,
        brightest_percentile=brightest_percentile,
        brightest_connected=brightest_connected,
    )
    skew = bragg_peak_skewnorm_from_centre(
        scattering_2d, q_map=d["q_map"], phi_map=d["phi_map"],
        cx=centre["cx"], cy=centre["cy"], region_mask=centre["region"], valid_mask=valid,
    )
    plot_brightest_pixel_qphi_overlay(
        scattering_2d, q_map=d["q_map"], phi_map=d["phi_map"], valid_mask=valid, cmap="magma",
        brightest_region_method=brightest_region_method,
        roi_half_size=roi_half_size,
        brightest_percentile=brightest_percentile,
        brightest_connected=brightest_connected,
    )
    print("Centre: cx=%.2f, cy=%.2f  q=%.5f  φ=%.5f rad" % (centre["cx"], centre["cy"], centre["q"], centre["phi_rad"]))
    print("Skew-normal (x, lineout): loc=%.2f scale=%.2f shape=%.3f skew=%.3f" % (skew["x_loc"], skew["x_scale"], skew["x_shape"], skew["x_skew"]))
    print("Skew-normal (y, lineout): loc=%.2f scale=%.2f shape=%.3f skew=%.3f" % (skew["y_loc"], skew["y_scale"], skew["y_shape"], skew["y_skew"]))
    print("Skew-normal (q, region):  loc=%.5f scale=%.5f shape=%.3f skew=%.3f" % (skew["q_loc"], skew["q_scale"], skew["q_shape"], skew["q_skew"]))
    print("Skew-normal (φ, region):  loc=%.5f rad scale=%.5f shape=%.3f skew=%.3f" % (skew["phi_loc"], skew["phi_scale"], skew["phi_shape"], skew["phi_skew"]))
    return {"centre": centre, "skewnorm": skew}


# ---------------------------------------------------------------------------
# Execution functions (call these from CLI or from if __name__ == "__main__")
# ---------------------------------------------------------------------------

def execute_find_bragg_peak_center_from_scattering_2d_with_overlay():
    (cy, cx), info = find_bragg_peak_center_from_scattering_2d_with_overlay_function(
        filename,
        bright_percentile=99.7,
        smooth_sigma_px=1.0,
        weight_mode="log",
    )
    print(f"Bragg peak centre: cy={cy:.2f}, cx={cx:.2f}")
    print(info)


def exec_bragg_peak_shape_metrics_fixed_q_phi(
    *,
    brightest_region_method: str = "roi_around_argmax",
    roi_half_size: int = 250,
    brightest_percentile: float = 90.0,
    brightest_connected: bool = True,
    use_skewnorm_fit: bool = True,
    refine_skewnorm_mle: bool = False,
    **kwargs,
):
    """Run Bragg peak shape metrics; pass options to override defaults."""
    with h5py.File(filename, "r") as f:
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
    npz_path = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")
    d = np.load(npz_path, allow_pickle=False)
    metrics = bragg_peak_shape_metrics_fixed_q_phi_from_maps(
        scattering_2d,
        q_map=d["q_map"],
        phi_map=d["phi_map"],
        valid_mask=d.get("valid_mask", None),
        hot_z_thresh=kwargs.pop("hot_z_thresh", 12.0),
        brightest_region_method=brightest_region_method,
        roi_half_size=roi_half_size,
        brightest_percentile=brightest_percentile,
        brightest_connected=brightest_connected,
        use_skewnorm_fit=use_skewnorm_fit,
        refine_skewnorm_mle=refine_skewnorm_mle,
        **kwargs,
    )
    print("Bragg peak shape metrics (fixed q/phi via per-pixel maps):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    plot_bragg_peak_shape_metrics_overlay_from_maps(
        scattering_2d,
        q_map=d["q_map"],
        phi_map=d["phi_map"],
        valid_mask=d.get("valid_mask", None),
        metrics=metrics,
        cmap="magma",
    )
    return metrics


def exec_make_and_save_inferred_qphi_maps():
    results = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_IPA_NBH_1_att0100_260K_001_results.hdf")
    out = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")
    save_inferred_qphi_maps_npz(results, out)


def exec_quick_check_inferred_qphi_npz():
    npz_path = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")
    d = np.load(npz_path, allow_pickle=False)
    q_map = d["q_map"]
    phi_map = d["phi_map"]
    Qx_map = d["Qx_map"]
    Qy_map = d["Qy_map"]
    valid = d["valid_mask"]
    print("Shapes:", "q_map:", q_map.shape, "phi_map:", phi_map.shape, "valid:", valid.shape)
    print("Valid fraction:", float(np.mean(valid)))
    vv = valid & np.isfinite(q_map) & np.isfinite(phi_map)
    print("Ranges on valid pixels:")
    print("  q:", float(np.nanmin(q_map[vv])), "to", float(np.nanmax(q_map[vv])))
    print("  phi(rad):", float(np.nanmin(phi_map[vv])), "to", float(np.nanmax(phi_map[vv])))
    print("  Qx:", float(np.nanmin(Qx_map[vv])), "to", float(np.nanmax(Qx_map[vv])))
    print("  Qy:", float(np.nanmin(Qy_map[vv])), "to", float(np.nanmax(Qy_map[vv])))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, arr, title in zip(axes, [q_map, phi_map, valid], ["q_map", "phi_map (rad)", "valid_mask"]):
        im = ax.imshow(arr, origin="lower", interpolation="nearest")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description="APS 08-ID-E XPCS analysis: inspection, Bragg peak metrics, q/φ maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s h5-inspector --file-id A073
  %(prog)s g2 --filename /path/to/results.hdf
  %(prog)s bragg-peak-metrics --file-id A073
        """,
    )
    parser.add_argument(
        "--file-id",
        default=DEFAULT_FILE_ID,
        help="Sample/file ID (e.g. A073). Used to find results HDF when --filename is not set (default: %(default)s).",
    )
    parser.add_argument(
        "--filename",
        type=Path,
        default=None,
        help="Path to results HDF. If set, overrides --file-id.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory to search for results when file-id is not in overrides (default: %s)."
        % DEFAULT_BASE_DIR,
    )

    sub = parser.add_subparsers(dest="command", required=True, help="Command to run")

    def add_file_options(p):
        """Add --file-id, --filename, --base-dir to a subparser so they work after the command."""
        p.add_argument("--file-id", default=DEFAULT_FILE_ID, help="Sample/file ID (default: %(default)s).")
        p.add_argument("--filename", type=Path, default=None, help="Path to results HDF (overrides --file-id).")
        p.add_argument("--base-dir", type=Path, default=None, help="Base dir to search for results HDF.")

    def add_file_cmd(name, help_text):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(needs_file=True)
        add_file_options(p)
        return p

    add_file_cmd("h5-inspector", "Print HDF5 structure of the results file.").set_defaults(
        run=lambda: h5_file_inspector(h5_file)
    )
    add_file_cmd("g2", "Plot g2 autocorrelation (pixelwise normalisation).").set_defaults(
        run=lambda: g2_plotter(h5_file)
    )
    add_file_cmd("ttc", "Plot two-time correlation matrix (single mask).").set_defaults(
        run=lambda: ttc_plotter(h5_file)
    )
    add_file_cmd("intensity-vs-time", "Plot spatial mean intensity vs time.").set_defaults(
        run=lambda: intensity_vs_time(h5_file)
    )
    add_file_cmd("static-vs-dynamic-bins", "Compare static vs dynamic bin integrated intensities.").set_defaults(
        run=lambda: static_vs_dynamic_bins(h5_file)
    )
    add_file_cmd("combined-plot", "Overview: ROI mask, g2, TTC grid, scattering crop.").set_defaults(
        run=lambda: combined_plot(h5_file)
    )
    add_file_cmd("q-spacing", "Print q and φ spacing (diffs).").set_defaults(
        run=lambda: q_spacing_inspector(h5_file)
    )
    add_file_cmd("integrated-intensities-inspector", "Load and plot scattering_1d, segments; print q/φ.").set_defaults(
        run=lambda: integrated_intensities_inspector(h5_file)
    )
    add_file_cmd("integrated-intensities-plot", "Plot mean and segment scattering_1d in (q, φ).").set_defaults(
        run=lambda: integrated_intensities_plot(h5_file)
    )
    add_file_cmd("bragg-peak-center", "Find Bragg peak centre (bright-region centroid) and show overlay.").set_defaults(
        run=execute_find_bragg_peak_center_from_scattering_2d_with_overlay
    )
    p_brightest = add_file_cmd("bragg-peak-brightest", "Centre of brightest region: cross + q, φ label.")
    p_brightest.add_argument("--brightest-percentile", type=float, default=90.0, dest="brightest_percentile", help="Percentile for brightest region (default: 90; higher = tighter, centre nearer peak).")
    p_brightest.add_argument("--brightest-region", choices=["roi_around_argmax", "percentile", "connected"], default="connected", dest="brightest_region_method", help="How to define brightest region (default: connected).")
    p_brightest.add_argument("--roi-half-size", type=int, default=250, dest="roi_half_size", help="Half-size of box in px for roi_around_argmax.")
    p_brightest.add_argument("--no-connected", action="store_false", dest="brightest_connected", help="With percentile: use all pixels above threshold.")
    p_brightest.set_defaults(run=exec_bragg_peak_brightest)
    p_skewnorm = add_file_cmd("bragg-peak-skewnorm", "Centre of brightest region + skew-normal fits (x,y lineouts; q,φ region).")
    p_skewnorm.add_argument("--brightest-percentile", type=float, default=90.0, dest="brightest_percentile", help="Percentile for brightest region (default: 90; higher = tighter).")
    p_skewnorm.add_argument("--brightest-region", choices=["roi_around_argmax", "percentile", "connected"], default="connected", dest="brightest_region_method", help="How to define brightest region (default: connected).")
    p_skewnorm.add_argument("--roi-half-size", type=int, default=250, dest="roi_half_size", help="Half-size of box in px for roi_around_argmax.")
    p_skewnorm.add_argument("--no-connected", action="store_false", dest="brightest_connected", help="With percentile: use all pixels above threshold.")
    p_skewnorm.set_defaults(run=exec_bragg_peak_skewnorm)
    p_bragg = add_file_cmd("bragg-peak-metrics", "Bragg peak shape metrics from per-pixel q/φ maps + overlay.")
    p_bragg.add_argument(
        "--brightest-region",
        choices=["roi_around_argmax", "percentile", "connected"],
        default="roi_around_argmax",
        dest="brightest_region_method",
        help="How to define the peak region (default: roi_around_argmax).",
    )
    p_bragg.add_argument("--roi-half-size", type=int, default=250, dest="roi_half_size", help="Half-size of box in px for roi_around_argmax.")
    p_bragg.add_argument("--brightest-percentile", type=float, default=90.0, dest="brightest_percentile", help="Percentile threshold for percentile/connected (default: 90).")
    p_bragg.add_argument("--no-connected", action="store_false", dest="brightest_connected", help="With percentile: use all pixels above threshold, not just connected component.")
    p_bragg.add_argument("--no-skewnorm", action="store_false", dest="use_skewnorm_fit", help="Use moment-based centre/skew instead of skew-normal fit.")
    p_bragg.add_argument("--refine-skewnorm-mle", action="store_true", dest="refine_skewnorm_mle", help="Refine skew-normal fit with weighted MLE.")
    p_bragg.set_defaults(run=exec_bragg_peak_shape_metrics_fixed_q_phi)
    add_file_cmd("make-qphi-maps", "Infer q/φ maps from static qmap and save to .npz.").set_defaults(
        run=exec_make_and_save_inferred_qphi_maps
    )
    add_file_cmd("check-qphi-npz", "Load and sanity-check inferred q/φ .npz; show q_map, phi_map, valid.").set_defaults(
        run=exec_quick_check_inferred_qphi_npz
    )

    sub.add_parser("oauth-test", help="Test Google OAuth (spreadsheet).").set_defaults(
        needs_file=False, run=oauth_test
    )
    sub.add_parser("image-upload", help="Upload image to Google Sheet (internal helper).").set_defaults(
        needs_file=False, run=image_upload
    )
    sub.add_parser("figure-upload", help="Upload a test figure to Google Sheet.").set_defaults(
        needs_file=False, run=figure_upload
    )

    return parser


def main():
    global filename, h5_file
    parser = _build_parser()
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if getattr(args, "needs_file", True):
        if args.filename is not None:
            filename = Path(args.filename)
        else:
            filename = resolve_results_path(args.file_id, args.base_dir)
        h5_file = filename
    else:
        filename = h5_file = None

    if args.command == "bragg-peak-metrics":
        exec_bragg_peak_shape_metrics_fixed_q_phi(
            brightest_region_method=getattr(args, "brightest_region_method", "roi_around_argmax"),
            roi_half_size=getattr(args, "roi_half_size", 250),
            brightest_percentile=getattr(args, "brightest_percentile", 90.0),
            brightest_connected=getattr(args, "brightest_connected", True),
            use_skewnorm_fit=getattr(args, "use_skewnorm_fit", True),
            refine_skewnorm_mle=getattr(args, "refine_skewnorm_mle", False),
        )
    elif args.command == "bragg-peak-brightest":
        exec_bragg_peak_brightest(
            brightest_region_method=getattr(args, "brightest_region_method", "connected"),
            roi_half_size=getattr(args, "roi_half_size", 250),
            brightest_percentile=getattr(args, "brightest_percentile", 90.0),
            brightest_connected=getattr(args, "brightest_connected", True),
        )
    elif args.command == "bragg-peak-skewnorm":
        exec_bragg_peak_skewnorm(
            brightest_region_method=getattr(args, "brightest_region_method", "connected"),
            roi_half_size=getattr(args, "roi_half_size", 250),
            brightest_percentile=getattr(args, "brightest_percentile", 90.0),
            brightest_connected=getattr(args, "brightest_connected", True),
        )
    else:
        args.run()


if __name__ == "__main__":
    main()