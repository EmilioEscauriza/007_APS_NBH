# ============================================================
# Google Sheets upload + XPCS plotting utilities (tidy version)
# ============================================================
#
# What this script does
# ---------------------
# 1) Looks up sample IDs for a given position name (e.g. "A5") in a Google Sheet
# 2) For each sample, finds the corresponding *_results.hdf in BASE_DIR
# 3) Generates overview / g2 / TTC grids (3x3 or 5x5 around the brightest mask)
# 4) Saves locally, optionally uploads PNGs to Google Sheets
# 5) Adds a NEW "q-dependent TTC" plot: TTC grid annotated with Δq (radial),
#    Δq_t (tangential), |Δq| ranges and corresponding real-space length-scale ranges
#
# Notes
# -----
# - No per-panel titles for TTC grids (keeps space). Everything goes in the textbox.
# - q and phi are read from:
#       q_list   = f["xpcs/qmap/dynamic_v_list_dim0"][...]
#       phi_list = f["xpcs/qmap/dynamic_v_list_dim1"][...]
#   These are axis lists (typically len(q_list)=10 rings, len(phi_list)=30 azimuth bins).
# - Mask index mapping assumes flattened index:
#       mask = iq*stride + iphi
#   where stride defaults to len(phi_list) (typically 30).
#
# ============================================================
# Imports
# ============================================================

import re
import time
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np

import matplotlib as mpl
mpl.use("macosx")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import gspread

import httplib2
from google_auth_httplib2 import AuthorizedHttp

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError


# ============================================================
# Small utilities
# ============================================================

def _as_1d(x) -> np.ndarray:
    return np.asarray(x).squeeze().reshape(-1)


def symmetrize_ttc(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    return C + C.T - np.diag(np.diag(C))


def clip_ttc(C: np.ndarray, p_hi: float = 99.9) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    m = np.isfinite(C)
    if not np.any(m):
        return C
    lo, hi = np.percentile(C[m], [0.0, float(p_hi)])
    return np.clip(C, lo, hi)


def execute_with_backoff(request, tries: int = 6, base_delay: float = 1.0):
    for attempt in range(tries):
        try:
            return request.execute()
        except (HttpError, ConnectionResetError, TimeoutError) as e:
            if attempt == tries - 1:
                raise
            sleep_s = base_delay * (2 ** attempt) + random.random()
            print(f"Upload failed ({type(e).__name__}), retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


# ============================================================
# Sheets scanning helpers
# ============================================================

def get_ids_for_position(ws, position_name: str, *, id_col: int = 1, position_col: int = 3, header_rows: int = 1):
    """
    Return a list of (row_number, sample_id) for rows where position_col == position_name.
    """
    col_ids = ws.col_values(id_col)
    col_pos = ws.col_values(position_col)

    n = min(len(col_ids), len(col_pos))
    out = []
    for i in range(header_rows, n):
        if col_pos[i].strip() == position_name:
            out.append((i + 1, col_ids[i].strip()))
    return out


def get_position_for_sample(ws, sample_id: str, *, id_col: int = 1, position_col: int = 3, header_rows: int = 1) -> str:
    """
    Look up the position name (e.g. 'A5') for a given sample_id (e.g. 'A013').
    """
    col_ids = ws.col_values(id_col)
    col_pos = ws.col_values(position_col)
    n = min(len(col_ids), len(col_pos))

    for i in range(header_rows, n):
        if col_ids[i].strip() == sample_id:
            return col_pos[i].strip()

    raise ValueError(f"Sample ID {sample_id} not found in spreadsheet")


def find_results_hdf(base_dir: Path, sample_id: str) -> Path | None:
    pattern = f"{sample_id}_*_results.hdf"
    matches = sorted(base_dir.glob(pattern))
    return matches[0] if matches else None


# ============================================================
# Google auth + API clients
# ============================================================

def get_creds(token_path: str, creds_path: str, scopes: list[str]) -> Credentials:
    creds = None
    if Path(token_path).exists():
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, scopes)
            creds = flow.run_local_server(port=0)
            Path(token_path).write_text(creds.to_json())

    return creds


def get_ws_and_drive(creds: Credentials, spreadsheet_id: str, tab_name: str):
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(tab_name)

    authed_http = AuthorizedHttp(creds, http=httplib2.Http(timeout=300))
    drive = build("drive", "v3", http=authed_http, cache_discovery=False)

    print("Opened spreadsheet:", sh.title, "| worksheet:", ws.title)
    return ws, drive


# ============================================================
# Upload + local save helpers
# ============================================================

def upload_fig_to_cell(
    ws,
    drive,
    fig,
    cell: str,
    upload_name: str,
    *,
    upload_folder_id: str,
    dpi: int = 300,
):
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)

        media = MediaIoBaseUpload(buf, mimetype="image/png", resumable=True)
        req = drive.files().create(
            body={"name": upload_name, "parents": [upload_folder_id]},
            media_body=media,
            fields="id",
        )
        created = execute_with_backoff(req)
        file_id = created["id"]

        drive.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

        image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
        formula = f'=IMAGE("{image_url}")'

        ws.update(
            range_name=cell,
            values=[[formula]],
            value_input_option="USER_ENTERED",
        )
        return file_id
    finally:
        buf.close()


def save_fig_local(fig, out_dir: Path, position_name: str, fig_key: str, sample_id: str, *, figtype_dir: dict, dpi: int):
    subdir = out_dir / position_name / figtype_dir[fig_key]
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"{sample_id}.png"
    fig.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight")
    return out_path


# ============================================================
# HDF loading + mask selection
# ============================================================

@dataclass(frozen=True)
class CommonData:
    roi_map: np.ndarray
    scat2d: np.ndarray
    g2: np.ndarray
    q_list: np.ndarray
    phi_list: np.ndarray
    stride: int


def load_common_data(hdf_path: Path) -> CommonData:
    with h5py.File(hdf_path, "r") as f:
        roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scat = f["xpcs/temporal_mean/scattering_2d"][...]
        scat2d = scat[0, :, :] if scat.ndim == 3 else scat

        g2 = f["xpcs/twotime/normalized_g2"][...]
        q_list = _as_1d(f["xpcs/qmap/dynamic_v_list_dim0"][...])
        phi_list = _as_1d(f["xpcs/qmap/dynamic_v_list_dim1"][...])

    stride = int(len(phi_list)) if len(phi_list) else 30
    return CommonData(roi_map=roi_map, scat2d=scat2d, g2=g2, q_list=q_list, phi_list=phi_list, stride=stride)


def load_c2_map(hdf_path: Path, mask_idx: int) -> np.ndarray:
    ttc_tree = f"xpcs/twotime/correlation_map/c2_00{int(mask_idx):03d}"
    with h5py.File(hdf_path, "r") as f:
        return f[ttc_tree][...]


def find_brightest_mask_by_integrated_intensity(roi_map: np.ndarray, scat2d: np.ndarray, *, n_masks: int = 300) -> int:
    """
    Find the mask label with the largest integrated intensity in scat2d.
    """
    roi_map = np.asarray(roi_map)
    scat2d = np.asarray(scat2d)

    intens = np.zeros(int(n_masks), dtype=np.float64)
    for mi in range(int(n_masks)):
        m = (roi_map == mi)
        if np.any(m):
            intens[mi] = float(np.nansum(scat2d[m]))
        else:
            intens[mi] = -np.inf

    return int(np.nanargmax(intens))


def neighborhood_offsets(n: int, *, stride: int) -> list[int]:
    """
    Offsets for an n×n neighborhood around a center index in a flattened (q,phi) grid.

    Ordering matches your process_position convention:
        [-29,   1,  31,
         -30,   0,  30,
         -31,  -1,  29]
    for n=3, stride=30.
    """
    if n % 2 == 0:
        raise ValueError("n must be odd (e.g., 3, 5).")

    r = n // 2
    offsets: list[int] = []
    for dx in range(r, -r - 1, -1):     # +phi to -phi (top to bottom)
        for dy in range(-r, r + 1):     # -q to +q (left to right)
            offsets.append(dy * int(stride) + dx)
    return offsets


def compute_neighborhood_indices(
    roi_map: np.ndarray,
    scat2d: np.ndarray,
    *,
    n_masks: int,
    grid_n: int,
    stride: int,
) -> tuple[int, list[int]]:
    """
    Returns (center_mask, idxs) where center_mask is the brightest mask.
    """
    center = find_brightest_mask_by_integrated_intensity(roi_map, scat2d, n_masks=n_masks)
    offs = neighborhood_offsets(grid_n, stride=stride)
    idxs = [center + o for o in offs]
    return center, idxs


# ============================================================
# q/phi geometry helpers for q-dependent TTC labels
# ============================================================

def qphi_for_mask(mask_idx: int, q_list: np.ndarray, phi_list: np.ndarray, *, stride: int) -> tuple[float, float, int, int]:
    """
    Flattened mask -> (iq, iphi) -> (q, phi).
    """
    stride = int(stride)
    iq = int(mask_idx) // stride
    iphi = int(mask_idx) % stride

    if iq < 0 or iq >= len(q_list):
        raise IndexError(f"iq={iq} out of range for q_list (len={len(q_list)})")
    if iphi < 0 or iphi >= len(phi_list):
        raise IndexError(f"iphi={iphi} out of range for phi_list (len={len(phi_list)})")

    return float(q_list[iq]), float(phi_list[iphi]), iq, iphi


def infer_steps_from_axis_lists(q_list: np.ndarray, phi_list: np.ndarray) -> tuple[float, float]:
    uq = np.sort(np.unique(np.asarray(q_list, float)))
    up = np.sort(np.unique(np.asarray(phi_list, float)))

    uq = uq[np.isfinite(uq)]
    up = up[np.isfinite(up)]

    if uq.size < 2:
        raise ValueError("Not enough q values to infer dq step.")
    if up.size < 2:
        raise ValueError("Not enough phi values to infer dphi step.")

    dq_step = float(np.median(np.diff(uq)))
    dphi_step_deg = float(np.median(np.diff(up)))
    return dq_step, dphi_step_deg


def _minmax_from_corners(xlo: float, xhi: float, ylo: float, yhi: float) -> tuple[float, float]:
    """
    Min/max of sqrt(x^2+y^2) over rectangle corners.
    """
    vals = [
        float(np.hypot(xlo, ylo)),
        float(np.hypot(xlo, yhi)),
        float(np.hypot(xhi, ylo)),
        float(np.hypot(xhi, yhi)),
    ]
    return min(vals), max(vals)


def length_scale_nm_from_qinvA(q_invA: float) -> float:
    """
    ℓ = 2π/q. Convert Å -> nm by *0.1.
    """
    q = float(q_invA)
    if not np.isfinite(q) or q <= 0:
        return np.nan
    return (2.0 * np.pi / q) * 0.1


def label_ranges_for_mask(
    mask_idx: int,
    *,
    center_mask: int,
    q_list: np.ndarray,
    phi_list: np.ndarray,
    stride: int,
    dq_step: float,
    dphi_step_deg: float,
) -> dict:
    """
    Build ranges for:
      - dq_radial in Å^-1 (relative to center)
      - dq_tangential in Å^-1 (≈ q0*Δphi_rad, relative to center)
      - |Δq| magnitude in Å^-1 (pythagorean, using corner extremes)
      - length-scale range nm via ℓ = 2π/|Δq|
    """
    q0, phi0, iq0, iphi0 = qphi_for_mask(center_mask, q_list, phi_list, stride=stride)
    q_m, phi_m, iq_m, iphi_m = qphi_for_mask(mask_idx, q_list, phi_list, stride=stride)

    # Center offsets in axis units
    dq_center = q_m - q0
    dphi_center_deg = phi_m - phi0

    # Each mask occupies a half-bin in q and phi
    dq_lo = dq_center - 0.5 * dq_step
    dq_hi = dq_center + 0.5 * dq_step

    dphi_lo_deg = dphi_center_deg - 0.5 * dphi_step_deg
    dphi_hi_deg = dphi_center_deg + 0.5 * dphi_step_deg

    # Tangential q shift: q0 * Δphi (radians)
    dqt_lo = q0 * np.deg2rad(dphi_lo_deg)
    dqt_hi = q0 * np.deg2rad(dphi_hi_deg)

    # magnitude of Δq over rectangle corners
    dmag_lo, dmag_hi = _minmax_from_corners(dq_lo, dq_hi, dqt_lo, dqt_hi)

    # Convert to length scales: ℓ = 2π/|Δq|
    # dmag_lo can be 0 near center => ℓ_hi = inf
    if dmag_hi <= 0 or not np.isfinite(dmag_hi):
        ell_lo_nm = np.nan
    else:
        ell_lo_nm = length_scale_nm_from_qinvA(dmag_hi)  # smallest length at largest |Δq|

    if dmag_lo <= 0 or not np.isfinite(dmag_lo):
        ell_hi_nm = np.inf
    else:
        ell_hi_nm = length_scale_nm_from_qinvA(dmag_lo)

    return dict(
        q0=q0,
        phi0=phi0,
        q=q_m,
        phi=phi_m,
        iq=iq_m,
        iphi=iphi_m,
        dq_lo=dq_lo, dq_hi=dq_hi,
        dqt_lo=dqt_lo, dqt_hi=dqt_hi,
        dmag_lo=dmag_lo, dmag_hi=dmag_hi,
        ell_lo_nm=ell_lo_nm, ell_hi_nm=ell_hi_nm,
    )


# ============================================================
# Plot builders
# ============================================================

def make_overview_fig(
    sample_id: str,
    roi_map: np.ndarray,
    scat2d: np.ndarray,
    idxs: list[int],
    *,
    title: str,
    half_crop: int = 200,
):
    combined_mask = np.isin(roi_map, idxs)

    fig, ax = plt.subplots()

    # Boost neighborhood masks
    I = scat2d.astype(float, copy=False).copy()
    I[combined_mask] *= 10.0

    # Crop around neighborhood centroid
    ys, xs = np.where(combined_mask)
    cy = int(np.round(ys.mean())) if ys.size else I.shape[0] // 2
    cx = int(np.round(xs.mean())) if xs.size else I.shape[1] // 2

    ymin = max(cy - half_crop, 0)
    ymax = min(cy + half_crop, I.shape[0])
    xmin = max(cx - half_crop, 0)
    xmax = min(cx + half_crop, I.shape[1])

    Icrop = I[ymin:ymax, xmin:xmax]
    Mcrop = roi_map[ymin:ymax, xmin:xmax]

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    Ishow = Icrop.copy()
    Ishow[Ishow <= 0] = 0.0
    Ishow_ma = np.ma.masked_less_equal(Ishow, 0.0)
    vmax = float(Ishow_ma.max()) if Ishow_ma.count() else 1.0

    im = ax.imshow(
        Ishow_ma,
        origin="upper",
        cmap=cmap,
        norm=LogNorm(vmin=0.1, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_facecolor("black")

    # Single-pass borders around masks in idxs
    M = Mcrop
    in_neigh = np.isin(M, idxs)

    boundary = np.zeros_like(M, dtype=bool)
    dv = (M[1:, :] != M[:-1, :])
    tv = in_neigh[1:, :] | in_neigh[:-1, :]
    boundary[1:, :] |= dv & tv

    dh = (M[:, 1:] != M[:, :-1])
    th = in_neigh[:, 1:] | in_neigh[:, :-1]
    boundary[:, 1:] |= dh & th

    overlay = np.zeros((boundary.shape[0], boundary.shape[1], 4), dtype=float)
    overlay[boundary] = (0.0, 0.0, 0.0, 1.0)  # black
    ax.imshow(overlay, origin="upper", interpolation="nearest")

    ax.set_title(f"{sample_id} {title}")
    ax.axis("off")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig


def make_g2s_fig(sample_id: str, g2: np.ndarray, idxs: list[int], *, title: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    x = np.arange(g2.shape[0])

    for mi in idxs:
        j = mi - 1  # your convention
        if 0 <= j < g2.shape[1]:
            ax.semilogx(x, g2[:, j], label=f"M{mi}")

    ax.set_title(f"{sample_id} {title}")
    ax.set_ylabel("g2(q,τ)")
    ax.set_xlabel("Delay time τ (index)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def make_twotime_grid_fig(
    sample_id: str,
    hdf_path: Path,
    idxs: list[int],
    *,
    grid_n: int,
    figsize: tuple[float, float],
    clip_hi_percentile: float = 99.9,
    textbox_fontsize: int = 10,
    suptitle: str = "",
):
    """
    Plain TTC grid with textbox per panel, no per-panel titles.
    """
    fig, axes = plt.subplots(grid_n, grid_n, figsize=figsize)
    axes = np.array(axes).reshape(grid_n, grid_n)

    for k, ax in enumerate(axes.flat):
        mi = int(idxs[k])
        C = load_c2_map(hdf_path, mi)
        C = symmetrize_ttc(C)
        Cplot = clip_ttc(C, p_hi=float(clip_hi_percentile))

        ax.imshow(Cplot, origin="lower", cmap="plasma", interpolation="nearest")
        ax.axis("off")

        txt = f"M{mi}\nmin={np.nanmin(C):.3g}\nmax={np.nanmax(C):.3g}"
        ax.text(
            0.04, 0.96,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=textbox_fontsize,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
        )

    if suptitle:
        fig.suptitle(f"{sample_id} {suptitle}", fontsize=18)
    fig.tight_layout()
    return fig


# ============================================================
# NEW: q-dependent TTC grid plot
# ============================================================

def plot_q_dependent_ttc(
    *,
    sample_id: str,
    base_dir: Path,
    out_dir: Path,
    ws,                      # used to map sample_id -> position folder structure
    grid_n: int = 5,          # 3 or 5
    n_masks: int = 300,
    clip_hi_percentile: float = 99.9,
    textbox_fontsize: int = 10,
    dpi: int = 200,
    out_name: str | None = None,
) -> Path:
    """
    Make a TTC grid (3x3 or 5x5) around the brightest mask, with textbox labels showing:
      - mask index
      - Δq_radial range [Å^-1]
      - Δq_tangential range [Å^-1]  (q0*Δphi)
      - |Δq| range [Å^-1] (pythagorean/corner bounds)
      - length-scale range [nm] via ℓ=2π/|Δq|
    """
    if grid_n not in (3, 5):
        raise ValueError("grid_n must be 3 or 5")

    position_name = get_position_for_sample(ws, sample_id)
    hdf_path = find_results_hdf(base_dir, sample_id)
    if hdf_path is None:
        raise FileNotFoundError(f"No results HDF found for {sample_id} in {base_dir}")

    common = load_common_data(hdf_path)
    stride = int(common.stride)

    # brightest mask + neighborhood (ordering matches process_position)
    center_mask, idxs = compute_neighborhood_indices(
        common.roi_map,
        common.scat2d,
        n_masks=n_masks,
        grid_n=grid_n,
        stride=stride,
    )

    # offset = -60
    # center_mask = center_mask + offset
    # new_idxs = [i + offset for i in idxs]
    # idxs = new_idxs
    # print(center_mask)
    # print(idxs)

    dq_step, dphi_step_deg = infer_steps_from_axis_lists(common.q_list, common.phi_list)

    # build grid
    fig, axes = plt.subplots(grid_n, grid_n, figsize=(12, 12) if grid_n == 5 else (7.2, 7.2))
    axes = np.array(axes).reshape(grid_n, grid_n)

    # Center mask q0/phi0 for reference (only used implicitly in labels)
    q0, phi0, *_ = qphi_for_mask(center_mask, common.q_list, common.phi_list, stride=stride)

    for k, ax in enumerate(axes.flat):
        mi = int(idxs[k])

        # load TTC
        C = load_c2_map(hdf_path, mi)
        C = symmetrize_ttc(C)
        Cplot = clip_ttc(C, p_hi=float(clip_hi_percentile))

        ax.imshow(Cplot, origin="lower", cmap="plasma", interpolation="nearest")
        ax.axis("off")

        # label ranges
        rr = label_ranges_for_mask(
            mi,
            center_mask=center_mask,
            q_list=common.q_list,
            phi_list=common.phi_list,
            stride=stride,
            dq_step=dq_step,
            dphi_step_deg=dphi_step_deg,
        )

        # pretty formatting
        def fmt_nm(x):
            if x == np.inf:
                return "∞"
            if not np.isfinite(x):
                return "nan"
            return f"{x:.3g}"

        txt = (
            f"q={rr['q']:.4g} Å⁻¹\n"
            f"φ={rr['phi']:.4g}°\n"
            f"ℓ=[{fmt_nm(rr['ell_lo_nm'])},{fmt_nm(rr['ell_hi_nm'])}]nm"
        )

        ax.text(
            0.04, 0.96,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=textbox_fontsize,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
        )

    # one small overall title only (optional). If you want *none*, comment this out.
    fig.suptitle(
        f"{sample_id} q-dependent TTC grid (center=M{center_mask}, q0={q0:.4g} Å⁻¹, φ0={phi0:.4g}°)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()

    # save
    out_base = out_dir / "q_dependent_ttc" / f"position_{position_name}" / sample_id
    out_base.mkdir(parents=True, exist_ok=True)

    if out_name is None:
        out_name = f"{sample_id}_qdep_ttc_ctx{grid_n}x{grid_n}.png"

    out_path = out_base / out_name
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.show()
    # plt.close(fig)

    return out_path


# ============================================================
# Plot selection / filtering helpers
# ============================================================

def normalize_keys(keys: Optional[Iterable[str]], all_keys: set[str]) -> set[str]:
    return set(all_keys) if keys is None else set(keys)


def should_generate(key: str, generate_keys: Optional[Iterable[str]], all_keys: set[str]) -> bool:
    return key in normalize_keys(generate_keys, all_keys)


def should_upload(
    key: str,
    upload_enabled: bool,
    upload_keys: Optional[Iterable[str]],
    generate_keys: Optional[Iterable[str]],
    all_keys: set[str],
) -> bool:
    if not upload_enabled:
        return False
    if not should_generate(key, generate_keys, all_keys):
        return False
    if upload_keys is None:
        return True
    return key in set(upload_keys)


# ============================================================
# Main processing pipeline
# ============================================================

def process_one_scan(
    sample_id: str,
    row: int,
    hdf_path: Path,
    ws,
    drive,
    *,
    position_name: str,
    out_dir: Path,
    generate_keys=None,
    upload_enabled: bool = True,
    upload_keys=None,
    # config dicts
    figtype_dir: dict[str, str],
    plot_cols: dict[str, str],
    dpi_by_plot: dict[str, int],
    all_plot_keys: set[str],
    upload_folder_id: str,
):
    gen = normalize_keys(generate_keys, all_plot_keys)
    common = load_common_data(hdf_path)

    need_9 = any(k in gen for k in ("overview_9", "g2s_9", "twotime_9"))
    need_25 = any(k in gen for k in ("overview_25", "g2s_25", "twotime_25"))

    idxs_9 = idxs_25 = None
    if need_9:
        _, idxs_9 = compute_neighborhood_indices(common.roi_map, common.scat2d, n_masks=300, grid_n=3, stride=common.stride)
    if need_25:
        _, idxs_25 = compute_neighborhood_indices(common.roi_map, common.scat2d, n_masks=300, grid_n=5, stride=common.stride)

    figs = {}

    # 9-mask set
    if "overview_9" in gen:
        figs["overview_9"] = make_overview_fig(sample_id, common.roi_map, common.scat2d, idxs_9, title="9-mask overview")
    if "g2s_9" in gen:
        figs["g2s_9"] = make_g2s_fig(sample_id, common.g2, idxs_9, title="9-mask g2")
    if "twotime_9" in gen:
        figs["twotime_9"] = make_twotime_grid_fig(
            sample_id, hdf_path, idxs_9,
            grid_n=3, figsize=(7, 7),
            suptitle="9-mask TTC",
            textbox_fontsize=10,
        )

    # 25-mask set
    if "overview_25" in gen:
        figs["overview_25"] = make_overview_fig(sample_id, common.roi_map, common.scat2d, idxs_25, title="25-mask overview")
    if "g2s_25" in gen:
        figs["g2s_25"] = make_g2s_fig(sample_id, common.g2, idxs_25, title="25-mask g2")
    if "twotime_25" in gen:
        figs["twotime_25"] = make_twotime_grid_fig(
            sample_id, hdf_path, idxs_25,
            grid_n=5, figsize=(12, 12),
            suptitle="25-mask TTC",
            textbox_fontsize=10,
        )

    try:
        for key, fig in figs.items():
            dpi = int(dpi_by_plot[key])

            # local save
            local_path = save_fig_local(fig, out_dir, position_name, key, sample_id, figtype_dir=figtype_dir, dpi=dpi)

            # upload
            if should_upload(key, upload_enabled, upload_keys, generate_keys, all_plot_keys):
                cell = f"{plot_cols[key]}{row}"
                upload_name = f"{sample_id}_{key}.png"
                upload_fig_to_cell(ws, drive, fig, cell, upload_name, upload_folder_id=upload_folder_id, dpi=dpi)
                print(f"Saved + uploaded: {local_path}  →  {ws.title}!{cell}")
            else:
                print(f"Saved local: {local_path}")

    finally:
        for fig in figs.values():
            plt.close(fig)


def process_position(
    position_name: str,
    base_dir: Path,
    ws,
    drive,
    *,
    out_dir: Path,
    generate_keys=None,
    upload_enabled: bool = True,
    upload_keys=None,
    start_sample_id: str | None = None,
    start_row: int | None = None,
    start_index: int = 0,
    # config dicts
    figtype_dir: dict[str, str],
    plot_cols: dict[str, str],
    dpi_by_plot: dict[str, int],
    all_plot_keys: set[str],
    upload_folder_id: str,
):
    rows_and_ids = get_ids_for_position(ws, position_name)
    print(f"Found {len(rows_and_ids)} scans at position {position_name}")

    # decide where to start
    if start_row is not None:
        rows_and_ids = [(r, sid) for (r, sid) in rows_and_ids if r >= start_row]
    elif start_sample_id is not None:
        start_pos = None
        for i, (r, sid) in enumerate(rows_and_ids):
            if sid == start_sample_id:
                start_pos = i
                break
        if start_pos is None:
            raise ValueError(f"start_sample_id={start_sample_id} not found in position {position_name}")
        rows_and_ids = rows_and_ids[start_pos:]
    else:
        rows_and_ids = rows_and_ids[start_index:]

    print(f"Starting from: {rows_and_ids[0] if rows_and_ids else 'nothing to do'}")

    for row, sample_id in rows_and_ids:
        hdf_path = find_results_hdf(base_dir, sample_id)
        if hdf_path is None:
            print(f"SKIP: no HDF file found for {sample_id}")
            continue

        print(f"Processing {sample_id} (row {row})")
        process_one_scan(
            sample_id=sample_id,
            row=row,
            hdf_path=hdf_path,
            ws=ws,
            drive=drive,
            position_name=position_name,
            out_dir=out_dir,
            generate_keys=generate_keys,
            upload_enabled=upload_enabled,
            upload_keys=upload_keys,
            figtype_dir=figtype_dir,
            plot_cols=plot_cols,
            dpi_by_plot=dpi_by_plot,
            all_plot_keys=all_plot_keys,
            upload_folder_id=upload_folder_id,
        )


# ============================================================
# plot_single_mask_scan (kept structure as you wanted)
# ============================================================

def plot_single_mask_scan(
    *,
    sample_id: str,
    mask_n: int,
    base_dir: Path,
    out_dir: Path,
    ws,                 # used to map sample_id -> position
    grid_n: int = 5,     # 3 or 5
    n_masks: int = 300,
    dpi: int = 250,
    figsize=(18, 5.5),
    stride: int | None = None,
    border_width: float = 1.5,
    half_crop: int = 220,
    out_name: str | None = None,
    # highlight controls
    neigh_boost: float = 10.0,
    other_dim: float = 0.35,
    highlight_boost: float = 25.0,
    highlight_outline: bool = True,
    outline_rgba=(1.0, 1.0, 1.0, 1.0),
):
    """
    One combined figure for a single scan + single mask:
      [overview with neighborhood borders (selected mask highlighted) | g2 for mask_n | TTC for mask_n]

    Saves to:
      out_dir/individual_scan_plots/position_<POS>/<SAMPLE_ID>/<SAMPLE_ID>_mask_<mask>_ctxNxN.png
    """
    if grid_n not in (3, 5):
        raise ValueError("grid_n must be 3 or 5")

    position_name = get_position_for_sample(ws, sample_id)

    hdf_path = find_results_hdf(base_dir, sample_id)
    if hdf_path is None:
        raise FileNotFoundError(f"No results HDF found for {sample_id} in {base_dir}")

    common = load_common_data(hdf_path)
    stride_eff = int(stride) if stride is not None else int(common.stride)

    # context neighborhood around brightest mask (for borders)
    _, idxs = compute_neighborhood_indices(common.roi_map, common.scat2d, n_masks=n_masks, grid_n=grid_n, stride=stride_eff)

    # load TTC for this mask
    C = load_c2_map(hdf_path, mask_n)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.10, 1.15], wspace=0.35)

    # ----------------------------
    # (1) Overview
    # ----------------------------
    ax0 = fig.add_subplot(gs[0])
    I = common.scat2d.astype(float, copy=False).copy()

    neigh = np.isin(common.roi_map, idxs)
    sel = (common.roi_map == mask_n)

    I[neigh] *= neigh_boost
    I[neigh & ~sel] *= other_dim
    I[sel] *= highlight_boost

    ys, xs = np.where(neigh)
    if ys.size == 0 or xs.size == 0:
        cy, cx = I.shape[0] // 2, I.shape[1] // 2
    else:
        cy = int(np.round(ys.mean()))
        cx = int(np.round(xs.mean()))

    ymin = max(cy - half_crop, 0)
    ymax = min(cy + half_crop, I.shape[0])
    xmin = max(cx - half_crop, 0)
    xmax = min(cx + half_crop, I.shape[1])

    Icrop = I[ymin:ymax, xmin:xmax]
    Mcrop = common.roi_map[ymin:ymax, xmin:xmax]

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    Ishow = np.ma.masked_less_equal(Icrop, 0.0)
    vmax = float(Ishow.max()) if Ishow.count() else 1.0

    im0 = ax0.imshow(
        Ishow,
        origin="upper",
        cmap=cmap,
        norm=LogNorm(vmin=0.1, vmax=vmax),
        interpolation="nearest",
    )
    ax0.set_facecolor("black")

    in_neigh = np.isin(Mcrop, idxs)
    boundary = np.zeros_like(Mcrop, dtype=bool)
    boundary[1:, :] |= (Mcrop[1:, :] != Mcrop[:-1, :]) & (in_neigh[1:, :] | in_neigh[:-1, :])
    boundary[:, 1:] |= (Mcrop[:, 1:] != Mcrop[:, :-1]) & (in_neigh[:, 1:] | in_neigh[:, :-1])

    if border_width and border_width > 1:
        N = int(round(border_width)) - 1
        b = boundary.copy()
        for _ in range(N):
            b2 = b.copy()
            b2[1:, :] |= b[:-1, :]
            b2[:-1, :] |= b[1:, :]
            b2[:, 1:] |= b[:, :-1]
            b2[:, :-1] |= b[:, 1:]
            b = b2
        boundary = b

    overlay = np.zeros((boundary.shape[0], boundary.shape[1], 4), dtype=float)
    overlay[boundary] = (0.0, 0.0, 0.0, 1.0)
    ax0.imshow(overlay, origin="upper", interpolation="nearest")

    if highlight_outline:
        sel_crop = (Mcrop == mask_n)
        sel_b = np.zeros_like(sel_crop, dtype=bool)
        sel_b[1:, :] |= (sel_crop[1:, :] != sel_crop[:-1, :])
        sel_b[:, 1:] |= (sel_crop[:, 1:] != sel_crop[:, :-1])

        if border_width and border_width > 1:
            N = int(round(border_width)) - 1
            b = sel_b.copy()
            for _ in range(N):
                b2 = b.copy()
                b2[1:, :] |= b[:-1, :]
                b2[:-1, :] |= b[1:, :]
                b2[:, 1:] |= b[:, :-1]
                b2[:, :-1] |= b[:, 1:]
                b = b2
            sel_b = b

        sel_overlay = np.zeros((sel_b.shape[0], sel_b.shape[1], 4), dtype=float)
        sel_overlay[sel_b] = outline_rgba
        ax0.imshow(sel_overlay, origin="upper", interpolation="nearest")

    ax0.set_title(f"{sample_id} overview (M{mask_n} highlighted, ctx {grid_n}×{grid_n})")
    ax0.axis("off")

    div0 = make_axes_locatable(ax0)
    cax0 = div0.append_axes("right", size="4%", pad=0.05)
    fig.colorbar(im0, cax=cax0)

    # ----------------------------
    # (2) g2 for mask_n
    # ----------------------------
    ax1 = fig.add_subplot(gs[1])
    tau = np.arange(common.g2.shape[0])
    j = mask_n - 1
    if 0 <= j < common.g2.shape[1]:
        ax1.semilogx(tau, common.g2[:, j], lw=2)
        # map mask to q/phi:
        try:
            q_m, phi_m, *_ = qphi_for_mask(mask_n, common.q_list, common.phi_list, stride=stride_eff)
            ax1.set_title(f"g2 for M{mask_n}\nq={q_m:.3f} Å⁻¹, φ={phi_m:.3f}°")
        except Exception:
            ax1.set_title(f"g2 for M{mask_n}")
    else:
        ax1.set_title(f"g2 for M{mask_n} (out of range)")

    ax1.set_xlabel("Delay time τ (index)")
    ax1.set_ylabel("g2(τ)", labelpad=10)
    ax1.grid(True, alpha=0.3)

    # ----------------------------
    # (3) TTC for mask_n
    # ----------------------------
    ax2 = fig.add_subplot(gs[2])
    C = symmetrize_ttc(C)
    Cplot = clip_ttc(C, p_hi=99.9)

    im2 = ax2.imshow(Cplot, origin="lower", cmap="plasma", interpolation="nearest")
    ax2.set_title(f"TTC for M{mask_n}")
    ax2.set_xlabel("t₁")
    ax2.set_ylabel("t₂")

    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="4%", pad=0.05)
    fig.colorbar(im2, cax=cax2)

    ax2.text(
        0.05, 0.95,
        f"M{mask_n}\nmin {np.nanmin(C):.2f}\nmax {np.nanmax(C):.2f}",
        transform=ax2.transAxes,
        ha="left", va="top",
        fontsize=12,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6, edgecolor="none"),
    )

    # save
    out_base = out_dir / "individual_scan_plots" / f"position_{position_name}" / sample_id
    out_base.mkdir(parents=True, exist_ok=True)

    if out_name is None:
        out_name = f"{sample_id}_mask_{mask_n:03d}_ctx{grid_n}x{grid_n}.png"

    out_path = out_base / out_name
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.show()
    # plt.close(fig)

    return out_path


# ============================================================
# Bragg peak center (argmax) + T from filename for t_dep_xrd
# ============================================================

def temperature_k_from_filename(hdf_path: Path) -> Optional[int]:
    """
    Extract temperature in Kelvin from results HDF filename (e.g. ..._260K_... → 260).
    """
    match = re.search(r"(\d+)K", hdf_path.name)
    return int(match.group(1)) if match else None


def bragg_peak_center_argmax(hdf_path: Path, *, phi_fast_axis: bool = True) -> tuple[float, float]:
    """
    Find Bragg peak (q0, phi0) from temporal-mean scattering_1d using argmax,
    same method as integrated_intensities_plot() in analysis_for_aps_08-ide-2025-1006.py.
    Returns (q0, phi0_deg).
    """
    with h5py.File(hdf_path, "r") as f:
        I1d = np.asarray(f["xpcs/temporal_mean/scattering_1d"][...])
        q = _as_1d(f["xpcs/qmap/static_v_list_dim0"][...])
        phi = _as_1d(f["xpcs/qmap/static_v_list_dim1"][...])
    nq, nphi = int(q.size), int(phi.size)
    if I1d.ndim != 2 or I1d.shape[0] != 1 or nq * nphi != int(I1d.shape[1]):
        raise ValueError(
            f"scattering_1d shape {I1d.shape} does not match q.size={nq} * phi.size={nphi}"
        )
    if phi_fast_axis:
        I_mean_qphi = I1d[0].reshape(nq, nphi)
    else:
        I_mean_phiq = I1d[0].reshape(nphi, nq)
        I_mean_qphi = np.transpose(I_mean_phiq, (1, 0))
    iq0, iphi0 = np.unravel_index(int(np.nanargmax(I_mean_qphi)), I_mean_qphi.shape)
    q0 = float(q[iq0])
    phi0 = float(phi[iphi0])
    return q0, phi0


def bragg_peak_center_argmax_second_moment_uncertainty(
    hdf_path: Path,
    *,
    phi_fast_axis: bool = True,
    eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    """
    Bragg peak (q0, phi0) from argmax; uncertainties from intensity-weighted second moments
    on the mean (q, φ) map. Center = argmax; σ_q, σ_φ = sqrt(weighted variance) with w = I.
    Returns (q0, phi0, sigma_q, sigma_phi).
    """
    with h5py.File(hdf_path, "r") as f:
        I1d = np.asarray(f["xpcs/temporal_mean/scattering_1d"][...])
        q = _as_1d(f["xpcs/qmap/static_v_list_dim0"][...])
        phi = _as_1d(f["xpcs/qmap/static_v_list_dim1"][...])
    nq, nphi = int(q.size), int(phi.size)
    if I1d.ndim != 2 or I1d.shape[0] != 1 or nq * nphi != int(I1d.shape[1]):
        raise ValueError(
            f"scattering_1d shape {I1d.shape} does not match q.size={nq} * phi.size={nphi}"
        )
    if phi_fast_axis:
        I_mean_qphi = I1d[0].reshape(nq, nphi)
    else:
        I_mean_phiq = I1d[0].reshape(nphi, nq)
        I_mean_qphi = np.transpose(I_mean_phiq, (1, 0))

    iq0, iphi0 = np.unravel_index(int(np.nanargmax(I_mean_qphi)), I_mean_qphi.shape)
    q0 = float(q[iq0])
    phi0 = float(phi[iphi0])

    w = np.clip(np.asarray(I_mean_qphi, dtype=np.float64), 0.0, None)
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= eps:
        uq = np.sort(np.unique(np.asarray(q, dtype=np.float64)))
        uphi = np.sort(np.unique(np.asarray(phi, dtype=np.float64)))
        dq = float(np.min(np.diff(uq))) / 2.0 if uq.size >= 2 else 1e-5
        dphi = float(np.min(np.diff(uphi))) / 2.0 if uphi.size >= 2 else 0.5
        return q0, phi0, dq, dphi

    q_2d = np.asarray(q, dtype=np.float64)[:, np.newaxis]  # (nq, nphi)
    phi_2d = np.asarray(phi, dtype=np.float64)[np.newaxis, :]  # (nq, nphi)
    q_bar = float(np.sum(w * q_2d) / sw)
    phi_bar = float(np.sum(w * phi_2d) / sw)
    var_q = float(np.sum(w * (q_2d - q_bar) ** 2) / sw)
    var_phi = float(np.sum(w * (phi_2d - phi_bar) ** 2) / sw)
    sigma_q = np.sqrt(max(var_q, 0.0))
    sigma_phi = np.sqrt(max(var_phi, 0.0))

    uq = np.sort(np.unique(np.asarray(q, dtype=np.float64)))
    uphi = np.sort(np.unique(np.asarray(phi, dtype=np.float64)))
    dq_half = float(np.min(np.diff(uq))) / 2.0 if uq.size >= 2 else 1e-5
    dphi_half = float(np.min(np.diff(uphi))) / 2.0 if uphi.size >= 2 else 0.5
    sigma_q = max(sigma_q, dq_half)
    sigma_phi = max(sigma_phi, dphi_half)
    return q0, phi0, sigma_q, sigma_phi


# ============================================================
# Execution functions
# ============================================================

def exec_google_sheet_upload():
    creds = get_creds(TOKEN_PATH, CREDS_PATH, SCOPES)
    ws, drive = get_ws_and_drive(creds, SPREADSHEET_ID, TAB_NAME)

    # Examples:
    # process_position("A6", BASE_DIR, ws, drive, out_dir=OUT_DIR, ...)
    # process_position(POSITION_NAME, BASE_DIR, ws, drive, out_dir=OUT_DIR, start_sample_id="A031", ...)

    process_position(
        POSITION_NAME,
        BASE_DIR,
        ws,
        drive,
        out_dir=OUT_DIR,
        generate_keys=GENERATE_KEYS,
        upload_enabled=UPLOAD_TO_SHEETS,
        upload_keys=UPLOAD_KEYS,
        figtype_dir=FIGTYPE_DIR,
        plot_cols=PLOT_COLS,
        dpi_by_plot=DPI_BY_PLOT,
        all_plot_keys=ALL_PLOT_KEYS,
        upload_folder_id=UPLOAD_FOLDER_ID,
    )


def exec_single_mask_plot_save():
    creds = get_creds(TOKEN_PATH, CREDS_PATH, SCOPES)
    ws, _drive = get_ws_and_drive(creds, SPREADSHEET_ID, TAB_NAME)

    plot_single_mask_scan(
        sample_id=SAMPLE_ID,
        mask_n=MASK_N,
        base_dir=BASE_DIR,
        out_dir=OUT_DIR,
        ws=ws,
        grid_n=5,
        border_width=1,
        dpi=250,
    )


def exec_q_dependent_ttc_plot():
    """
    New entrypoint for the q-dependent TTC grid.
    """
    creds = get_creds(TOKEN_PATH, CREDS_PATH, SCOPES)
    ws, _drive = get_ws_and_drive(creds, SPREADSHEET_ID, TAB_NAME)

    out_path = plot_q_dependent_ttc(
        sample_id=SAMPLE_ID,
        base_dir=BASE_DIR,
        out_dir=OUT_DIR,
        ws=ws,
        grid_n=5,              # 3 or 5
        n_masks=300,
        clip_hi_percentile=99.9,
        textbox_fontsize=10,
        dpi=200,
    )
    print("Saved:", out_path)


def t_dep_xrd_argmax(
    *,
    position_name: str | None = None,
    base_dir: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """
    For all samples at a position: find Bragg peak (q0, phi0) via argmax on scattering_1d,
    uncertainties from intensity-weighted second moments (σ_q, σ_φ), get temperature
    from the results HDF filename, then plot q0 and phi0 vs T with error bars.
    Uses module CONFIG (POSITION_NAME, BASE_DIR, OUT_DIR) when arguments are None.
    """
    base_dir = base_dir or BASE_DIR
    position_name = position_name or POSITION_NAME
    out_dir = out_dir or OUT_DIR

    creds = get_creds(TOKEN_PATH, CREDS_PATH, SCOPES)
    ws, _ = get_ws_and_drive(creds, SPREADSHEET_ID, TAB_NAME)
    rows_and_ids = get_ids_for_position(ws, position_name)
    if not rows_and_ids:
        print(f"No sample IDs found for position {position_name}")
        return

    T_list: list[int] = []
    q0_list: list[float] = []
    phi0_list: list[float] = []
    q0_err_list: list[float] = []
    phi0_err_list: list[float] = []
    sample_ids_used: list[str] = []

    for _row, sample_id in rows_and_ids:
        hdf_path = find_results_hdf(base_dir, sample_id)
        if hdf_path is None:
            print(f"SKIP: no HDF for {sample_id}")
            continue
        T = temperature_k_from_filename(hdf_path)
        if T is None:
            print(f"SKIP: no temperature in filename for {sample_id} ({hdf_path.name})")
            continue
        try:
            q0, phi0, q0_err, phi0_err = bragg_peak_center_argmax_second_moment_uncertainty(hdf_path)
        except Exception as e:
            print(f"SKIP: {sample_id} bragg_peak_center_argmax_second_moment_uncertainty failed: {e}")
            continue
        T_list.append(T)
        q0_list.append(q0)
        phi0_list.append(phi0)
        q0_err_list.append(q0_err)
        phi0_err_list.append(phi0_err)
        sample_ids_used.append(sample_id)

    if not T_list:
        print("No data to plot.")
        return

    # sort by T for clean curve
    order = np.argsort(T_list)
    T_arr = np.array(T_list)[order]
    q0_arr = np.array(q0_list)[order]
    phi0_arr = np.array(phi0_list)[order]
    q0_err_arr = np.array(q0_err_list)[order]
    phi0_err_arr = np.array(phi0_err_list)[order]

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.errorbar(T_arr[0:-1], q0_arr[0:-1], yerr=q0_err_arr[0:-1], fmt="o-", color="C0", capsize=3)
    ax1.set_ylabel("q₀ (Å⁻¹)")
    ax1.set_xlabel("T (K)")
    ax1.set_title("Bragg peak q vs temperature")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(T_arr[0:-1], phi0_arr[0:-1], yerr=phi0_err_arr[0:-1], fmt="s-", color="C1", capsize=3)
    ax2.set_ylabel("φ₀ (deg)")
    ax2.set_xlabel("T (K)")
    ax2.set_title("Bragg peak φ vs temperature")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Position {position_name}", y=0.98)
    fig.tight_layout()

    out_base = out_dir / "t_dep_xrd"
    out_base.mkdir(parents=True, exist_ok=True)
    out_path = out_base / f"bragg_vs_T_{position_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved:", out_path)

    plt.show()


# ============================================================
# CONFIG
# ============================================================

SPREADSHEET_ID = "1OAA7H4I3cgas32aSZkrLB8TOKHymMAv2uk_0eTywWcQ"
TAB_NAME = "IPA NBH"
TOKEN_PATH = "token.json"
CREDS_PATH = "client_secret_180145739842-0ug37lsh4qltki62e8te8bqkde9u25jb.apps.googleusercontent.com.json"

UPLOAD_FOLDER_ID = "18IccfznbNEgAewkGqAmXwhaLA9yTrYa-"
OUT_DIR = Path("//Users/emilioescauriza/Documents/repos/006_APS_8IDE/emilio_scripts/figures_export")

FIGTYPE_DIR = {
    "overview_9": "9_mask_overview",
    "g2s_9": "9_mask_g2",
    "twotime_9": "9_mask_twotime",
    "overview_25": "25_mask_overview",
    "g2s_25": "25_mask_g2",
    "twotime_25": "25_mask_twotime",
}

PLOT_COLS = {
    "overview_9": "AJ",
    "g2s_9": "AK",
    "twotime_9": "AL",
    "overview_25": "AM",
    "g2s_25": "AN",
    "twotime_25": "AO",
}

ALL_PLOT_KEYS = {
    "overview_9", "g2s_9", "twotime_9",
    "overview_25", "g2s_25", "twotime_25",
}

DPI_BY_PLOT = {
    "overview_9": 300,
    "g2s_9": 300,
    "twotime_9": 150,
    "overview_25": 250,
    "g2s_25": 250,
    "twotime_25": 80,
}

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

BASE_DIR = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")
# BASE_DIR = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01")
POSITION_NAME = "A4"
SAMPLE_ID = "A073"
MASK_N = 146

# Which plots to generate in process_position (None = all 6)
GENERATE_KEYS = None # {"twotime_9", "twotime_25"}  # None or for example: {"overview_25", "twotime_25"} to only generate these two
UPLOAD_TO_SHEETS = True  # True of False
UPLOAD_KEYS = None  # or example: {"overview_25"} to upload only overview_25 only but still generate others


if __name__ == "__main__":

    # exec_google_sheet_upload()
    # exec_single_mask_plot_save()
    # exec_q_dependent_ttc_plot()
    t_dep_xrd_argmax()
    # all_bragg_peaks_centre_plot()

    pass