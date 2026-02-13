import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from networkx.algorithms.operators.binary import difference

# from erik_file_transfer.notebooks.xpcs_1 import intensity
from source_functions import *
from xpcs import *
from sims import *
# from autocorrelations import *
import cv2
from scipy.special import erfinv
import pyopencl as cl
from pathlib import Path
import json
from scipy.fft import fft, ifft, fftfreq


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
    C = C + C.T - np.diag(np.diag(C))
    # renomalize C
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

    scattering_2d_reshape = scattering_2d[0, :, :]

    individual_mask_intensity = []


    print(dynamic_roi_map.shape)
    print(scattering_2d.shape)
    print(scattering_2d_reshape.shape)

    for i in range(300):
        individual_mask = dynamic_roi_map.copy()
        individual_mask[individual_mask != i+1] = 0
        individual_mask[individual_mask != 0] = 1
        scattering_2d_masked = scattering_2d_reshape * individual_mask
        individual_mask_intensity.append(np.sum(scattering_2d_masked))

    # individual_mask = dynamic_roi_map.copy()
    # individual_mask[individual_mask != 100] = 0
    # individual_mask[individual_mask != 0] = 1
    # scattering_2d_masked = scattering_2d_reshape * individual_mask
    # individual_mask_intensity.append(np.sum(scattering_2d_masked))
    #
    # individual_mask_intensity = np.array(individual_mask_intensity)

    plt.figure()
    # plt.plot(individual_mask_intensity)
    plt.semilogy(np.arange(1, 301, 1), individual_mask_intensity)
    plt.xlabel('mask number')
    plt.ylabel('integrated intensity')


    # plt.figure()
    # plt.imshow(individual_mask)

    plt.figure()
    plt.imshow(dynamic_roi_map)

    plt.figure()
    plt.imshow(individual_mask)

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
        ttc = f["xpcs/twotime/correlation_map/c2_00194"][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]
        q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]

    run_name = os.path.basename(h5_file).split("_")[0]


    scattering_2d_reshape = scattering_2d[0, :, :]
    individual_mask_intensity = []

    print('q:', q)

    print('phi:', phi)

    for index in np.arange(0, 300, 1):
        # print('index:', index, ', q:', q[10-int(np.floor(index/10))], ', phi:', phi[int(np.floor(index/30))])
        print('index:', index, ', x:', int((index // 30)), ', q:', q[int((index // 30))],
              ', y:', int(index % 30), ', phi:', phi[int(index % 30)])

    individual_mask = dynamic_roi_map.copy()
    # individual_mask[(individual_mask != 165) & (individual_mask != 225)] = 0
    individual_mask[individual_mask != 135] = 0
    individual_mask[individual_mask != 0] = 1

    for i in range(300):
        individual_mask = dynamic_roi_map.copy()
        individual_mask[individual_mask != i] = 0
        individual_mask[individual_mask != 0] = 1
        scattering_2d_masked = scattering_2d_reshape * individual_mask
        individual_mask_intensity.append(np.sum(scattering_2d_masked))

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
    x = np.arange(len(g2[:, 100]))
    # plt.errorbar(delay[0, :], G2_result, yerr=G2_error, fmt='none', ecolor='b', capsize=2)
    for i in idxs:
        plt.semilogx(x, g2[:, i-1], label='M' + str(i) + ', q='+f"{q[int((i // 30))]:.3f}"
                                        + ',  phi='+f"{phi[int(i % 30)]:.3f}")
    plt.title('g2 autocorrelation for experiment ' + run_name)
    plt.ylabel('g2(q,tau)')
    plt.xlabel('Delay Time, tau')
    plt.legend()

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    for i, ax in enumerate(axes.flat):
        path = f"xpcs/twotime/correlation_map/c2_00{idxs[i]:03d}"
        with h5py.File(filename, "r") as f:
            C = f[path][...]
        C = C + C.T - np.diag(np.diag(C))
        # renomalize C
        # C = C - np.min(C)
        # C = C / np.max(C)
        lo, hi = np.percentile(C, [0, 99.9])
        C = np.clip(C, lo, hi)
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
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        ttc = f["xpcs/twotime/correlation_map/c2_00194"][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]
        q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]

    run_name = os.path.basename(h5_file).split("_")[0]


    scattering_2d_reshape = scattering_2d[0, :, :]
    individual_mask_intensity = []

    print('q:', q)

    print('phi:', phi)

    for i in range(9):
        print(q[i + 1] - q[i])

    for j in range(29):
        print(phi[j + 1] - phi[j])

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

def bragg_peak_centroid_and_skew_qphi(
    I_mean_qphi: np.ndarray,
    q: np.ndarray,
    phi_deg: np.ndarray,
    *,
    eps: float = 1e-12,
    roi_mask_qphi: np.ndarray | None = None,  # optional (nq,nphi) boolean ROI
) -> dict:
    """
    Intensity-weighted centroid and skewness in q and phi.

    Notes
    -----
    - q: linear stats
    - phi: circular mean, then skewness of wrapped residuals dphi in [-180,180)
    """
    I = np.asarray(I_mean_qphi, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    phi_deg = np.asarray(phi_deg, dtype=np.float64)

    nq, nphi = I.shape
    if q.size != nq or phi_deg.size != nphi:
        raise ValueError(f"Shape mismatch: I={I.shape}, q={q.shape}, phi={phi_deg.shape}")

    if roi_mask_qphi is None:
        vv = np.isfinite(I) & (I > 0)
    else:
        vv = np.asarray(roi_mask_qphi, dtype=bool) & np.isfinite(I) & (I > 0)

    if not np.any(vv):
        raise RuntimeError("No valid pixels for centroid/skewness.")

    w = I[vv]
    sw = float(np.sum(w))
    if sw <= eps:
        raise RuntimeError("Sum of weights is zero.")

    # Build coordinate grids for the valid points
    iq, ip = np.nonzero(vv)
    qv = q[iq]                 # (N,)
    ph_deg = phi_deg[ip]       # (N,)

    # ---------- q mean + skew ----------
    q_mean = float(np.sum(w * qv) / sw)
    dq = qv - q_mean
    mu2_q = float(np.sum(w * dq * dq) / sw)
    mu3_q = float(np.sum(w * dq * dq * dq) / sw)
    sigma_q = float(np.sqrt(max(mu2_q, 0.0)))
    skew_q = float(mu3_q / (mu2_q ** 1.5 + eps))

    # ---------- phi circular mean ----------
    ph_rad = np.deg2rad(ph_deg)
    c = float(np.sum(w * np.cos(ph_rad)) / sw)
    s = float(np.sum(w * np.sin(ph_rad)) / sw)
    phi_mean_rad = float(np.arctan2(s, c))
    phi_mean_deg = float(np.rad2deg(phi_mean_rad))

    # ---------- phi skewness on wrapped residuals ----------
    dphi_deg = ph_deg - phi_mean_deg
    dphi_deg = (dphi_deg + 180.0) % 360.0 - 180.0  # wrap to [-180,180)

    mu2_phi = float(np.sum(w * dphi_deg * dphi_deg) / sw)
    mu3_phi = float(np.sum(w * dphi_deg * dphi_deg * dphi_deg) / sw)
    sigma_phi_deg = float(np.sqrt(max(mu2_phi, 0.0)))
    skew_phi = float(mu3_phi / (mu2_phi ** 1.5 + eps))

    return {
        "q_mean": q_mean,
        "phi_mean_deg": phi_mean_deg,
        "sigma_q": sigma_q,
        "sigma_phi_deg": sigma_phi_deg,
        "skew_q": skew_q,
        "skew_phi": skew_phi,
        "n_points": int(w.size),
    }

def integrated_intensities_plot(
    h5_file: str | Path,
    *,
    phi_fast_axis: bool = True,
    map_scale: str = "log",          # "linear" or "log"
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.8,
    relstd_vmax: float | None = None,
):
    """
    Plots scattering_1d (mean) and scattering_1d_segments (10 time segments).

    Flattening assumption:
      If phi_fast_axis=True (default):
          flat index = iq * nphi + iphi  -> reshape (nq, nphi)
      If phi_fast_axis=False:
          flat index = iphi * nq + iq    -> reshape (nphi, nq) then transpose to (nq, nphi)
    """
    h5_file = str(h5_file)

    import numpy as np
    import h5py
    import matplotlib.pyplot as plt

    with h5py.File(h5_file, "r") as f:
        I1d = f["xpcs/temporal_mean/scattering_1d"][...]
        Iseg = f["xpcs/temporal_mean/scattering_1d_segments"][...]
        q = f["xpcs/qmap/static_v_list_dim0"][...]
        phi = f["xpcs/qmap/static_v_list_dim1"][...]

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
    if phi_fast_axis:
        I_mean_qphi = I1d[0].reshape(nq, nphi)
        I_seg_qphi = Iseg.reshape(Iseg.shape[0], nq, nphi)  # (nseg, nq, nphi)
    else:
        # flat index = iphi*nq + iq
        I_mean_phiq = I1d[0].reshape(nphi, nq)
        I_seg_phiq = Iseg.reshape(Iseg.shape[0], nphi, nq)
        I_mean_qphi = np.transpose(I_mean_phiq, (1, 0))
        I_seg_qphi = np.transpose(I_seg_phiq, (0, 2, 1))

    # ----------------------------
    # Bragg peak center in (q, phi)
    # ----------------------------
    eps = 1e-12

    # weights, clip negatives just in case
    W = np.clip(I_mean_qphi, 0.0, None)
    sw = float(np.sum(W))

    if not np.isfinite(sw) or sw <= eps:
        raise RuntimeError("No positive intensity in I_mean_qphi to compute Bragg center.")

    # (1) argmax center
    iq_max, iphi_max = np.unravel_index(int(np.argmax(I_mean_qphi)), I_mean_qphi.shape)
    q_max = float(q[iq_max])
    phi_max = float(phi[iphi_max])  # degrees

    # (2) intensity-weighted centroid
    # weighted mean in q
    q_mean = float(np.sum(W * q[:, None]) / sw)

    # weighted circular mean in phi (degrees -> radians for trig)
    phi_rad = np.deg2rad(phi.astype(np.float64))
    c = float(np.sum(W * np.cos(phi_rad)[None, :]) / sw)
    s = float(np.sum(W * np.sin(phi_rad)[None, :]) / sw)
    phi_mean_rad = float(np.arctan2(s, c))
    phi_mean_deg = float((np.rad2deg(phi_mean_rad) + 180.0) % 360.0 - 180.0)  # wrap to [-180, 180)

    # nearest bin to centroid (useful for overlay and sanity)
    iq_c = int(np.argmin(np.abs(q - q_mean)))
    # circular distance in degrees for nearest phi
    dphi = (phi - phi_mean_deg + 180.0) % 360.0 - 180.0
    iphi_c = int(np.argmin(np.abs(dphi)))
    q_cent_bin = float(q[iq_c])
    phi_cent_bin = float(phi[iphi_c])

    print("Bragg peak center estimates from scattering_1d:")
    print(f"  argmax bin: iq={iq_max}, iphi={iphi_max}, q={q_max:.6f}, phi={phi_max:.3f} deg")
    print(f"  centroid:   q_mean={q_mean:.6f}, phi_mean={phi_mean_deg:.3f} deg")
    print(f"  nearest bin to centroid: iq={iq_c}, iphi={iphi_c}, q={q_cent_bin:.6f}, phi={phi_cent_bin:.3f} deg")

    metrics = bragg_peak_centroid_and_skew_qphi(I_mean_qphi, q, phi)
    print("Bragg peak centroid/skew from scattering_1d:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # representative phi index: closest to 0°
    iphi0 = int(np.argmin(np.abs(phi - 0.0)))

    # ---- summaries ----
    Iseg_q = I_seg_qphi.mean(axis=2)          # (nseg, nq)
    Imean_q = I_mean_qphi.mean(axis=1)        # (nq,)
    I_std_qphi = I_seg_qphi.std(axis=0)       # (nq, nphi)
    I_relstd_qphi = I_std_qphi / np.maximum(I_mean_qphi, 1e-12)

    # ---- display transforms + robust limits ----
    def _disp(arr: np.ndarray) -> np.ndarray:
        if map_scale.lower() == "log":
            return np.log10(np.clip(arr, 0.0, None) + 1e-12)
        return arr

    mean_disp = _disp(I_mean_qphi)
    rel_disp = I_relstd_qphi  # keep linear (already a ratio)

    # robust vmin/vmax for the mean map
    finite_mean = mean_disp[np.isfinite(mean_disp)]
    vmin = float(np.percentile(finite_mean, vmin_pct))
    vmax = float(np.percentile(finite_mean, vmax_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(finite_mean)), float(np.nanmax(finite_mean))

    # robust vmax for relstd
    finite_rel = rel_disp[np.isfinite(rel_disp)]
    if relstd_vmax is None:
        rel_vmax = float(np.percentile(finite_rel, 99.5))
    else:
        rel_vmax = float(relstd_vmax)

    # ---- plotting ----
    fig = plt.figure(figsize=(14.5, 8))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.28)

    # (A) Mean map in (q, phi)
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        mean_disp,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[phi.min(), phi.max(), q.min(), q.max()],
        vmin=vmin,
        vmax=vmax,
    )
    ax0.set_title(f"Mean scattering_1d → (q, φ)  [{map_scale}]")
    ax0.set_xlabel("φ (deg)")
    ax0.set_ylabel("q (Å$^{-1}$)")
    cblabel0 = "log10(Intensity + eps)" if map_scale.lower() == "log" else "Intensity (a.u.)"
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03, label=cblabel0)

    # (B) Segment evolution as (segment index, q) for φ≈0 slice
    ax1 = fig.add_subplot(gs[0, 1])
    seg_vs_q_phi0 = _disp(I_seg_qphi[:, :, iphi0])  # display same scaling
    finite_seg = seg_vs_q_phi0[np.isfinite(seg_vs_q_phi0)]
    svmin = float(np.percentile(finite_seg, vmin_pct))
    svmax = float(np.percentile(finite_seg, vmax_pct))
    im1 = ax1.imshow(
        seg_vs_q_phi0,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[q.min(), q.max(), 0, seg_vs_q_phi0.shape[0] - 1],
        vmin=svmin,
        vmax=svmax,
    )
    ax1.set_title(f"Segments vs q at φ≈{phi[iphi0]:.3f}° (closest to 0°)")
    ax1.set_xlabel("q (Å$^{-1}$)")
    ax1.set_ylabel("segment index (0..9)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03, label=cblabel0)

    # (C) φ-averaged intensity vs q for each segment + mean
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
        rel_disp,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[phi.min(), phi.max(), q.min(), q.max()],
        vmin=0.0,
        vmax=rel_vmax,
    )
    ax3.set_title("Temporal variability: std(seg)/mean  (q, φ)")
    ax3.set_xlabel("φ (deg)")
    ax3.set_ylabel("q (Å$^{-1}$)")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03, label="Relative std")

    fig.suptitle(f"Integrated intensity diagnostics\n{h5_file}", y=0.98, fontsize=12)
    # plt.show()


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

def execute_find_bragg_peak_center_from_scattering_2d_with_overlay():

    (cy, cx), info = find_bragg_peak_center_from_scattering_2d_with_overlay_function(
        filename,
        bright_percentile=99.7,
        smooth_sigma_px=1.0,
        weight_mode="log",
    )

    print(f"Bragg peak centre: cy={cy:.2f}, cx={cx:.2f}")
    print(info)

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


def make_phi_band_mask(
    phi_map: np.ndarray,
    *,
    phi0: float,
    dphi: float,
    valid: np.ndarray | None = None,
    q_map: np.ndarray | None = None,
    qmin: float | None = None,
    qmax: float | None = None,
) -> np.ndarray:
    """
    Boolean mask for pixels with phi within [phi0-dphi, phi0+dphi], with wraparound.
    Optionally also apply a q-range cut.
    """
    phi = np.asarray(phi_map, dtype=np.float64)
    if valid is None:
        vv = np.isfinite(phi)
    else:
        vv = np.asarray(valid, dtype=bool) & np.isfinite(phi)

    # wrap delta-phi to [-pi, pi]
    d = phi - float(phi0)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi

    m = vv & (np.abs(d) <= float(dphi))

    if q_map is not None and (qmin is not None or qmax is not None):
        q = np.asarray(q_map, dtype=np.float64)
        if qmin is not None:
            m &= (q >= float(qmin))
        if qmax is not None:
            m &= (q <= float(qmax))

    return m

def build_q_phi_maps_from_static_qmap(f, *, use_index_mapping: bool = True):
    """
    Build per-pixel Q_map and Phi_map (same shape as detector image)
    from the *static* qmap products in the results file.

    Assumes:
      - static_roi_map values: 0..(n_q*n_phi), where 0 means background
      - q_vals length = n_q, phi_vals length = n_phi
      - bins packed with phi fastest: flat = q_i*n_phi + phi_i
      - static_index_mapping is a length-(n_q*n_phi) permutation (optional)
    """
    roi_map = f["xpcs/qmap/static_roi_map"][...]
    idx_map = f["xpcs/qmap/static_index_mapping"][...]
    q_vals  = f["xpcs/qmap/static_v_list_dim0"][...]
    phi_vals = f["xpcs/qmap/static_v_list_dim1"][...]

    roi_map = np.asarray(roi_map)
    idx_map = np.asarray(idx_map, dtype=np.int64)
    q_vals = np.asarray(q_vals, dtype=np.float64)
    phi_vals = np.asarray(phi_vals, dtype=np.float64)

    n_q = int(q_vals.size)
    n_phi = int(phi_vals.size)
    n_bins = n_q * n_phi  # should be 3600

    # valid pixels are labeled 1..n_bins (0 = background)
    valid = (roi_map > 0) & (roi_map <= n_bins)

    Q_map = np.full(roi_map.shape, np.nan, dtype=np.float64)
    Phi_map = np.full(roi_map.shape, np.nan, dtype=np.float64)

    if not np.any(valid):
        return Q_map, Phi_map

    # 1..n_bins -> 0..n_bins-1
    roi0 = roi_map[valid].astype(np.int64) - 1

    if use_index_mapping:
        flat = idx_map[roi0]   # permutation into 0..n_bins-1
    else:
        flat = roi0            # assume roi labels already match flat packing

    # unpack (phi fastest)
    q_i = flat // n_phi
    p_i = flat %  n_phi

    Q_map[valid] = q_vals[q_i]
    Phi_map[valid] = phi_vals[p_i]
    return Q_map, Phi_map

def overlay_mask_contour(ax, mask: np.ndarray, *, color="lime", lw=2.0, alpha=0.9, zorder=10):
    """
    Draw contour boundary of a boolean mask on an existing imshow axes.
    Works like your ROI contours.
    """
    m = np.asarray(mask, dtype=bool)
    ax.contour(m.astype(np.float32), levels=[0.5], colors=[color], linewidths=float(lw),
               alpha=float(alpha), zorder=int(zorder))

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
    crop_half_size: int = 250,
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

    vv = valid  # or your ROI mask if you have one
    w = np.clip(img[vv], 0.0, None)

    iy, ix = np.nonzero(vv)
    sw = float(np.sum(w))
    if sw <= 0:
        raise RuntimeError("No positive weight for centroid.")

    cx = float(np.sum(w * ix) / sw)
    cy = float(np.sum(w * iy) / sw)

    half = crop_half_size  # 500x500 box

    H, W = img.shape
    x0 = int(max(cx - half, 0))
    x1 = int(min(cx + half, W))
    y0 = int(max(cy - half, 0))
    y1 = int(min(cy + half, H))

    img_c = img[y0:y1, x0:x1]
    q_c = q_map[y0:y1, x0:x1]
    phi_c = phi_map[y0:y1, x0:x1]
    valid_c = valid[y0:y1, x0:x1]

    extent = [x0, x1, y1, y0]

    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.2))
    im = ax.imshow(
        img_c,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
        extent=extent,
    )

    ax.set_title("Detector-space scattering_2d with q/phi contour overlay")
    ax.set_facecolor("black")
    ax.set_xlabel("Detector x (pixels)")
    ax.set_ylabel("Detector y (pixels)")

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

    # --- contour on CROPPED maps, using the same extent as imshow ---
    # flip vertically to match imshow(origin="upper")
    q_for_contour_c = np.where(valid_c, q_c, np.nan)
    ph_for_contour_c = np.where(valid_c, phi_c, np.nan)

    q_for_contour_c = np.flipud(q_for_contour_c)
    ph_for_contour_c = np.flipud(ph_for_contour_c)

    ax.contour(
        q_for_contour_c,
        levels=q_levels,
        linewidths=0.7,
        alpha=0.8,
        extent=extent,
    )

    ax.contour(
        ph_for_contour_c,
        levels=ph_levels,
        linewidths=0.7,
        alpha=0.5,
        extent=extent,
    )

    # --- lock the view to the crop (contours can otherwise autoscale) ---
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # because origin="upper"

    # Mark the mean (q_mean, phi_mean) by finding nearest pixel in (q,phi) space
    q_mean = float(metrics.get("q_mean"))
    phi_mean = float(metrics.get("phi_mean_rad"))

    ax.plot([cx], [cy], marker="x", markersize=10, mew=2)
    ax.text(
        cx + 50,
        cy + 50,
        f"Pixel centroid (x̄,ȳ)\nq̄={q_mean:.4f}\nφ̄={phi_mean:.4f} rad",
        fontsize=12,
        color="yellow",
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="log10(Intensity + eps)")
    fig.tight_layout()
    plt.show()

    return {"center_px_rc": (float(cy), float(cx))}

def regrid_detector_to_qphi(
    img: np.ndarray,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    q_edges: np.ndarray | None = None,
    phi_edges: np.ndarray | None = None,
    n_q: int = 400,
    n_phi: int = 360,
    phi_wrap: str = "pi",   # "pi" -> [-pi,pi), "2pi" -> [0,2pi)
    statistic: str = "mean" # "mean" or "median"
) -> dict:
    """
    Regrid detector-space intensity img(y,x) onto a regular (q, phi) grid using binning.

    Returns a dict with q_centers, phi_centers, I_qphi, N_qphi, edges.
    Shapes: I_qphi and N_qphi are (Nphi, Nq).
    """
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[0]
    img = np.asarray(img, dtype=np.float64)

    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)

    if img.shape != q_map.shape or img.shape != phi_map.shape:
        raise ValueError(f"Shape mismatch: img={img.shape}, q_map={q_map.shape}, phi_map={phi_map.shape}")

    if valid_mask is None:
        vv = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        vv = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)

    if not np.any(vv):
        raise RuntimeError("No valid pixels to regrid.")

    qv = q_map[vv].ravel()
    phv = phi_map[vv].ravel()
    Iv = img[vv].ravel()

    # Wrap phi
    if phi_wrap == "pi":
        phv = (phv + np.pi) % (2.0 * np.pi) - np.pi
        default_phi_lo, default_phi_hi = -np.pi, np.pi
    elif phi_wrap == "2pi":
        phv = phv % (2.0 * np.pi)
        default_phi_lo, default_phi_hi = 0.0, 2.0 * np.pi
    else:
        raise ValueError("phi_wrap must be 'pi' or '2pi'")

    # Build edges if not provided
    if q_edges is None:
        q_lo, q_hi = np.nanpercentile(qv, [0.5, 99.5])
        q_edges = np.linspace(float(q_lo), float(q_hi), int(n_q) + 1)

    if phi_edges is None:
        # You can also restrict phi range to the detector-covered region if you want
        phi_edges = np.linspace(float(default_phi_lo), float(default_phi_hi), int(n_phi) + 1)

    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    # Bin indices
    iq = np.searchsorted(q_edges, qv, side="right") - 1
    ip = np.searchsorted(phi_edges, phv, side="right") - 1

    Nq = len(q_centers)
    Np = len(phi_centers)

    in_range = (iq >= 0) & (iq < Nq) & (ip >= 0) & (ip < Np)
    iq = iq[in_range]
    ip = ip[in_range]
    Iv = Iv[in_range]

    # Accumulate
    N_qphi = np.zeros((Np, Nq), dtype=np.int64)

    if statistic == "mean":
        S_qphi = np.zeros((Np, Nq), dtype=np.float64)
        np.add.at(S_qphi, (ip, iq), Iv)
        np.add.at(N_qphi, (ip, iq), 1)
        I_qphi = np.full((Np, Nq), np.nan, dtype=np.float64)
        m = N_qphi > 0
        I_qphi[m] = S_qphi[m] / N_qphi[m]

    elif statistic == "median":
        # Median needs storing lists per bin (slower but robust to hot pixels)
        bins = [[[] for _ in range(Nq)] for __ in range(Np)]
        for p, q, val in zip(ip, iq, Iv):
            bins[p][q].append(float(val))
        I_qphi = np.full((Np, Nq), np.nan, dtype=np.float64)
        for p in range(Np):
            for q in range(Nq):
                if bins[p][q]:
                    arr = np.asarray(bins[p][q], dtype=np.float64)
                    I_qphi[p, q] = float(np.median(arr))
                    N_qphi[p, q] = int(arr.size)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")

    return {
        "q_edges": q_edges,
        "phi_edges": phi_edges,
        "q_centers": q_centers,
        "phi_centers": phi_centers,
        "I_qphi": I_qphi,     # shape (Nphi, Nq)
        "N_qphi": N_qphi,     # shape (Nphi, Nq)
    }

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

def bragg_peak_shape_metrics_fixed_q_phi_from_maps(
    scattering_2d: np.ndarray,
    *,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    hot_z_thresh: float = 12.0,
    eps: float = 1e-12,
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
        This only rejects extreme outliers, it does NOT mask long physical tails.

    Returns
    -------
    dict of scalar metrics (means, sigmas, skewness in q and phi).
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

    # freeze the mask used for pixel-space moments
    vv_used = vv.copy()

    iy0, ix0 = np.unravel_index(np.nanargmax(img), img.shape)

    half_w = 250  # tune
    half_h = 250  # tune
    roi = np.zeros_like(vv_used, dtype=bool)
    y0, x0 = iy0, ix0
    roi[max(0, y0 - half_h):min(img.shape[0], y0 + half_h + 1),
    max(0, x0 - half_w):min(img.shape[1], x0 + half_w + 1)] = True

    vv_used = vv_used & roi

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

    # skewness
    skew_x = np.sum(w * (ix - x_mean) ** 3) / (w_sum * sigma_x ** 3)
    skew_y = np.sum(w * (iy - y_mean) ** 3) / (w_sum * sigma_y ** 3)

    H, W = img.shape
    print("argmax array (x,y):", ix0, iy0, " display-y:", (H - 1) - iy0)
    print("centroid array (x,y):", x_mean, y_mean, " display-y:", (H - 1) - y_mean)

    return {
        "x_mean_px": float(x_mean),
        "y_mean_px": float(y_mean),
        "sigma_x_px": float(sigma_x),
        "sigma_y_px": float(sigma_y),
        "skew_x": float(skew_x),
        "skew_y": float(skew_y),
        "q_mean": q_mean,
        "phi_mean_rad": phi_mean,
        "sigma_q": sigma_q,
        "sigma_phi_rad": sigma_phi,
        "skew_q": skew_q,
        "skew_phi": skew_phi,
        "n_pix_used": n_pix,
        "frac_valid_used": frac_kept,
        "peak_intensity": peak_adu,
        "hot_z_thresh": float(hot_z_thresh),
    }



def exec_bragg_peak_shape_metrics_fixed_q_phi():

    with h5py.File(filename, "r") as f:
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        q = f["xpcs/qmap/static_v_list_dim0"][...]
        phi = f["xpcs/qmap/static_v_list_dim1"][...]

    # Example: point this to the file you just saved
    npz_path = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")
    d = np.load(npz_path, allow_pickle=False)

    metrics = bragg_peak_shape_metrics_fixed_q_phi_from_maps(
        scattering_2d,
        q_map=d["q_map"],
        phi_map=d["phi_map"],
        valid_mask=d.get("valid_mask", None),
        hot_z_thresh=12.0,
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
        crop_half_size=500,
    )

    return metrics


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

def exec_make_and_save_inferred_qphi_maps():
    results = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_IPA_NBH_1_att0100_260K_001_results.hdf")
    out = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")
    save_inferred_qphi_maps_npz(results, out)

def exec_quick_check_inferred_qphi_npz():

    npz_path = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")

    d = np.load(Path(npz_path), allow_pickle=False)

    q_map = d["q_map"]
    phi_map = d["phi_map"]
    Qx_map = d["Qx_map"]
    Qy_map = d["Qy_map"]
    valid = d["valid_mask"]

    print("Shapes:")
    print("  q_map:", q_map.shape, "phi_map:", phi_map.shape, "valid:", valid.shape)
    print("Valid fraction:", float(np.mean(valid)))

    print("Ranges on valid pixels:")
    vv = valid & np.isfinite(q_map) & np.isfinite(phi_map)
    print("  q:", float(np.nanmin(q_map[vv])), "to", float(np.nanmax(q_map[vv])))
    print("  phi(rad):", float(np.nanmin(phi_map[vv])), "to", float(np.nanmax(phi_map[vv])))
    print("  Qx:", float(np.nanmin(Qx_map[vv])), "to", float(np.nanmax(Qx_map[vv])))
    print("  Qy:", float(np.nanmin(Qy_map[vv])), "to", float(np.nanmax(Qy_map[vv])))

    # quick visual
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax0, ax1, ax2 = axes
    im0 = ax0.imshow(q_map, origin="upper", interpolation="nearest")
    ax0.set_title("q_map")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)

    im1 = ax1.imshow(phi_map, origin="upper", interpolation="nearest")
    ax1.set_title("phi_map (rad)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)

    im2 = ax2.imshow(valid, origin="upper", interpolation="nearest")
    ax2.set_title("valid_mask")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)

    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    plt.show()

def exec_build_q_phi_map():

    with h5py.File(h5_file, "r") as f:
        with h5py.File(h5_file, "r") as f:
            Q_full, Phi_full = build_q_phi_maps_from_geometry(f)

    print(Q_full)
    print(Phi_full)
    print(np.shape(Q_full))
    print(np.shape(Phi_full))

    # -----------------------------
    # Visualization
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(
        Q_full,
        origin="upper",
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].set_title("Q map (per pixel)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03, label="q")

    im1 = axes[1].imshow(
        Phi_full,
        origin="upper",
        cmap="twilight",
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title("Phi map (per pixel)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03, label="phi (deg)")

    fig.suptitle("Full detector q / φ maps (nearest-valid extrapolated)", fontsize=14)
    fig.tight_layout()
    plt.show()

def _wrap_deg_to_180(phi_deg: np.ndarray) -> np.ndarray:
    """Wrap degrees to [-180, 180)."""
    return (phi_deg + 180.0) % 360.0 - 180.0


def _circular_mean_deg(phi_deg: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> float:
    """Weighted circular mean in degrees, returns value in [-180, 180)."""
    phi_rad = np.deg2rad(phi_deg)
    sw = float(np.sum(w)) + eps
    c = float(np.sum(w * np.cos(phi_rad)) / sw)
    s = float(np.sum(w * np.sin(phi_rad)) / sw)
    mean_rad = float(np.arctan2(s, c))
    return float(_wrap_deg_to_180(np.rad2deg(mean_rad)))


def _circular_centered_deg(phi_deg: np.ndarray, phi0_deg: float) -> np.ndarray:
    """
    Return signed angular difference (deg) from phi0, wrapped to [-180, 180).
    """
    return _wrap_deg_to_180(phi_deg - float(phi0_deg))


def _weighted_moments_1d(x: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> tuple[float, float, float]:
    """
    Weighted mean, sigma, skewness for 1D variable x.
    Skewness is central mu3 / mu2^(3/2).
    """
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= eps:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.sum(w * x) / sw)
    dx = x - mean
    mu2 = float(np.sum(w * dx * dx) / sw)
    mu3 = float(np.sum(w * dx * dx * dx) / sw)

    sigma = float(np.sqrt(max(mu2, 0.0)))
    skew = float(mu3 / (mu2 ** 1.5 + eps))
    return mean, sigma, skew


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    sa = float(np.linalg.norm(a))
    sb = float(np.linalg.norm(b))
    if sa <= eps or sb <= eps:
        return float("nan")
    return float(np.dot(a, b) / (sa * sb))

def _peak_anchored_tail_asymmetry_1d(x: np.ndarray, w: np.ndarray, i0: int, p: float = 1.0, eps: float = 1e-12):
    """
    Peak-anchored left/right tail asymmetry about the ARGMAX index i0.

    A(p) = (L - R) / (L + R), where
      L = sum_{i<i0} w_i * |x_i - x0|^p
      R = sum_{i>i0} w_i * |x_i - x0|^p

    Negative => heavier/brighter tail on the LEFT (smaller x).
    Positive => heavier/brighter tail on the RIGHT (larger x).
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, None)

    if not (0 <= int(i0) < x.size):
        raise ValueError("i0 out of range")

    x0 = float(x[int(i0)])

    left = slice(0, int(i0))
    right = slice(int(i0) + 1, x.size)

    dxL = np.abs(x[left] - x0)
    dxR = np.abs(x[right] - x0)

    L = float(np.sum(w[left] * (dxL ** float(p))))
    R = float(np.sum(w[right] * (dxR ** float(p))))

    A = (R - L) / (L + R + float(eps))

    print("tail debug:",
          "x0=", x0,
          "L=", L, "R=", R,
          "A(L-R)=", (L - R) / (L + R + eps),
          "A(R-L)=", (R - L) / (L + R + eps))

    return A, L, R, x0


def _argmax_in_roi(I_qphi: np.ndarray, roi_qphi: np.ndarray) -> tuple[int, int]:
    """
    Return (iq0, iphi0) of the maximum intensity inside ROI.
    """
    M = np.where(roi_qphi, I_qphi, -np.inf)
    flat = int(np.argmax(M))
    iq0, iphi0 = np.unravel_index(flat, I_qphi.shape)
    return int(iq0), int(iphi0)

def integrated_intensities_peak_stability(
    h5_file: str | Path,
    *,
    q_range: tuple[float, float] = (1.09, 1.13),
    phi_range_deg: tuple[float, float] = (-10.0, 10.0),
    use_log_for_similarity: bool = True,
    log_eps: float = 1e-12,
    weight_power: float = 1.0,
    show_plots: bool = True,
) -> dict:
    """
    Peak + stability metrics using scattering_1d and scattering_1d_segments (10 segments).

    Physics ROI:
      - q within q_range
      - phi within phi_range_deg  (phi is in degrees)

    Metrics per segment (in ROI):
      - q_mean, sigma_q, skew_q
      - phi_mean_deg (circular), sigma_phi_deg, skew_phi
      - I_tot, I_peak

    Stability vs mean pattern (in ROI):
      - cosine_similarity[s]
      - pearson_r[s]  (on log(I+eps) if use_log_for_similarity=True)

    Returns a dict with arrays and a few summary scalars.
    """
    h5_file = str(h5_file)

    with h5py.File(h5_file, "r") as f:
        I1d = np.asarray(f["xpcs/temporal_mean/scattering_1d"][...], dtype=np.float64)
        Iseg = np.asarray(f["xpcs/temporal_mean/scattering_1d_segments"][...], dtype=np.float64)
        q = np.asarray(f["xpcs/qmap/static_v_list_dim0"][...], dtype=np.float64)
        phi = np.asarray(f["xpcs/qmap/static_v_list_dim1"][...], dtype=np.float64)

    # ---- sanity ----
    if I1d.ndim != 2 or I1d.shape[0] != 1:
        raise ValueError(f"Expected scattering_1d shape (1, 3600), got {I1d.shape}")
    if Iseg.ndim != 2 or Iseg.shape[1] != I1d.shape[1]:
        raise ValueError(f"Expected scattering_1d_segments shape (10, 3600), got {Iseg.shape}")

    nq = int(q.size)
    nphi = int(phi.size)
    if nq * nphi != int(I1d.shape[1]):
        raise ValueError(f"q.size * phi.size = {nq*nphi} does not match scattering_1d length {I1d.shape[1]}")

    # ---- reshape (phi fast axis) ----
    I_mean_qphi = I1d[0].reshape(nq, nphi)
    I_mean_phiq = I_mean_qphi.T
    I_seg_qphi = Iseg.reshape(Iseg.shape[0], nq, nphi)  # (nseg, nq, nphi)
    nseg = int(I_seg_qphi.shape[0])

    I_q = I_mean_qphi.mean(axis=1)  # or sum(axis=1), depending on what you plot
    # print("argmax iq:", np.argmax(I_q), "q at argmax:", q[np.argmax(I_q)])
    # print("left edge q:", q[0], "right edge q:", q[-1])

    I_q = I_mean_qphi.mean(axis=1)  # or sum(axis=1)
    iq0 = int(np.argmax(I_q))

    # print("peak:", iq0, q[iq0], I_q[iq0])
    # print("left  (iq0-5..iq0-1):", list(zip(range(iq0 - 5, iq0), q[iq0 - 5:iq0], I_q[iq0 - 5:iq0])))
    # print("right (iq0+1..iq0+5):", list(zip(range(iq0 + 1, iq0 + 6), q[iq0 + 1:iq0 + 6], I_q[iq0 + 1:iq0 + 6])))

    # compare_q_skew_two_methods(
    #     I_mean_qphi,
    #     I_seg_qphi,
    #     q,
    #     phi,
    #     choose_phi="argmax",  # or "closest0"
    #     # iphi=24,             # optional explicit override
    # )

    # ---- build ROI mask in (q,phi) ----
    q_lo, q_hi = float(min(q_range)), float(max(q_range))
    ph_lo, ph_hi = float(min(phi_range_deg)), float(max(phi_range_deg))

    qq, pp = np.meshgrid(q, phi, indexing="ij")  # (nq,nphi)
    roi = (qq >= q_lo) & (qq <= q_hi) & (pp >= ph_lo) & (pp <= ph_hi)

    if not np.any(roi):
        raise RuntimeError("ROI is empty. Check q_range and phi_range_deg against q,phi arrays.")

    # ----------------------------
    # Peak-anchored tail "skew" (Method A analog) for q and phi
    # ----------------------------
    # Anchor: argmax bin in the MEAN map, restricted to ROI
    M = np.where(roi, I_mean_qphi, -np.inf)
    iq0, iphi0 = np.unravel_index(int(np.argmax(M)), M.shape)

    # arrays to plot in ax4
    q_skew_peak = np.full((nseg,), np.nan, dtype=np.float64)
    phi_skew_peak = np.full((nseg,), np.nan, dtype=np.float64)

    # choose p=1.0 (your earlier output used p=1 as the intuitive tail metric)
    p_tail = 1.0

    for s in range(nseg):
        Is = I_seg_qphi[s]

        # q tail: single-phi lineout at iphi0
        wq = np.clip(Is[:, iphi0].astype(np.float64), 0.0, None)
        Aq, _, _, _ = _peak_anchored_tail_asymmetry_1d(q, wq, int(iq0), p=p_tail, eps=log_eps)
        q_skew_peak[s] = Aq

        # phi tail: single-q lineout at iq0
        wphi = np.clip(Is[int(iq0), :].astype(np.float64), 0.0, None)
        Aphi, _, _, _ = _peak_anchored_tail_asymmetry_1d(phi, wphi, int(iphi0), p=p_tail, eps=log_eps)
        phi_skew_peak[s] = Aphi

    # ---- per-segment metrics ----
    q_mean = np.full((nseg,), np.nan, dtype=np.float64)
    q_sigma = np.full((nseg,), np.nan, dtype=np.float64)
    q_skew = np.full((nseg,), np.nan, dtype=np.float64)

    phi_mean = np.full((nseg,), np.nan, dtype=np.float64)       # degrees
    phi_sigma = np.full((nseg,), np.nan, dtype=np.float64)      # degrees
    phi_skew = np.full((nseg,), np.nan, dtype=np.float64)

    I_tot = np.full((nseg,), np.nan, dtype=np.float64)
    I_peak = np.full((nseg,), np.nan, dtype=np.float64)

    # Flattened coords inside ROI
    q_roi = qq[roi].astype(np.float64)
    phi_roi = pp[roi].astype(np.float64)

    # Reference pattern for similarity
    I_roi_stack = np.empty((nseg, q_roi.size), dtype=np.float64)

    for s in range(nseg):
        Is = I_seg_qphi[s]
        w = Is[roi].astype(np.float64)

        # weights: non-negative, optionally emphasize peak
        w = np.clip(w, 0.0, None)
        if weight_power != 1.0:
            w = w ** float(weight_power)

        sw = float(np.sum(w))
        I_tot[s] = sw
        I_peak[s] = float(np.max(Is[roi]))

        # q moments
        qm, qs, qk = _weighted_moments_1d(q_roi, w)
        q_mean[s], q_sigma[s], q_skew[s] = qm, qs, qk

        # phi circular mean, then centered moments on wrapped differences
        phm = _circular_mean_deg(phi_roi, w)
        phi_mean[s] = phm
        dphi = _circular_centered_deg(phi_roi, phm)
        ph_mu, ph_sig, ph_sk = _weighted_moments_1d(dphi, w)
        phi_sigma[s] = ph_sig
        phi_skew[s] = ph_sk

        # store pattern for similarity (use raw intensity, similarity step decides log or not)
        I_roi_stack[s, :] = np.clip(Is[roi].astype(np.float64), 0.0, None)

    # ---- stability vs mean pattern ----
    I_ref = np.mean(I_roi_stack, axis=0)

    cos_sim = np.full((nseg,), np.nan, dtype=np.float64)
    pearson_r = np.full((nseg,), np.nan, dtype=np.float64)

    if use_log_for_similarity:
        Aref = np.log10(I_ref + float(log_eps))
    else:
        Aref = I_ref

    for s in range(nseg):
        A = np.log10(I_roi_stack[s] + float(log_eps)) if use_log_for_similarity else I_roi_stack[s]
        cos_sim[s] = _cosine_similarity(A, Aref)
        pearson_r[s] = _pearson_corr(A, Aref)

    # ---- summary scalars ----
    drift_q = float(np.nanmax(q_mean) - np.nanmin(q_mean))
    drift_phi = float(_wrap_deg_to_180(float(np.nanmax(phi_mean) - np.nanmin(phi_mean))))  # rough wrap-aware

    out = {
        "q": q,
        "phi_deg": phi,
        "q_range": (q_lo, q_hi),
        "phi_range_deg": (ph_lo, ph_hi),
        "roi_mask_qphi": roi,
        "I_mean_qphi": I_mean_qphi,
        "I_seg_qphi": I_seg_qphi,

        "q_mean": q_mean,
        "sigma_q": q_sigma,
        "skew_q": q_skew,

        "phi_mean_deg": phi_mean,
        "sigma_phi_deg": phi_sigma,
        "skew_phi": phi_skew,

        "I_tot": I_tot,
        "I_peak": I_peak,

        "cosine_similarity": cos_sim,
        "pearson_r": pearson_r,

        "drift_q": drift_q,
        "drift_phi_deg": drift_phi,

        "q_skew_peak": q_skew_peak,
        "phi_skew_peak": phi_skew_peak,
    }

    if show_plots:
        # ROI mean map
        mean_roi = np.where(roi, I_mean_qphi, np.nan)

        # Mean centroid (from mean map, in ROI)
        w0 = np.clip(I_mean_qphi[roi].astype(np.float64), 0.0, None)
        if weight_power != 1.0:
            w0 = w0 ** float(weight_power)
        q0m, _, _ = _weighted_moments_1d(q_roi, w0)
        ph0m = _circular_mean_deg(phi_roi, w0)

        fig = plt.figure(figsize=(14.5, 8))
        gs = fig.add_gridspec(2, 3, wspace=0.5, hspace=0.5)

        # (A) Mean map, with ROI boundary idea and centroid marker
        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(
            np.log10(np.clip(I_mean_phiq, 0.0, None) + float(log_eps)),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[q.min(), q.max(), phi.min(), phi.max()],
        )
        # ax0.axvline(ph_lo, lw=1.0, alpha=0.8)
        # ax0.axvline(ph_hi, lw=1.0, alpha=0.8)
        # ax0.axhline(q_lo, lw=1.0, alpha=0.8)
        # ax0.axhline(q_hi, lw=1.0, alpha=0.8)
        # ax0.plot([ph0m], [q0m], marker="x", ms=10, mew=2)
        ax0.set_title("Mean log10 intensity map (q, φ)\nROI bounds + mean centroid")
        ax0.set_ylabel("φ (deg)")
        ax0.set_xlabel("q (Å$^{-1}$)")
        fig.colorbar(im0, ax=ax0, fraction=0.04, pad=0.03, label="log10(I + eps)")

        # print(q[:5], q[-5:])

        # (B) Segment centroids drift in (q,phi)
        ax1 = fig.add_subplot(gs[0, 1])
        sc = ax1.scatter(phi_mean, q_mean, c=np.arange(nseg), s=60)
        ax1.plot(phi_mean, q_mean, lw=1.2, alpha=0.7)
        ax1.set_title("Segment centroid drift (q_mean vs φ_mean)")
        ax1.set_xlabel("φ_mean (deg)")
        ax1.set_ylabel("q_mean (Å$^{-1}$)")
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        # ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax1.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03, label="segment index")

        # (C) q_mean and phi_mean vs segment index
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(np.arange(nseg), q_mean, marker="o", lw=1.8, label="q_mean")
        ax2b = ax2.twinx()
        ax2b.plot(np.arange(nseg), phi_mean, marker="s", lw=1.6, alpha=0.85, label="phi_mean", linestyle="--")
        ax2.set_title("Centroid vs segment index")
        ax2.set_xlabel("segment index")
        ax2.set_ylabel("q_mean (Å$^{-1}$)")
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        # ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax2b.set_ylabel("φ_mean (deg)")
        ax2.grid(True, alpha=0.25)

        # (D) widths and skewness
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(np.arange(nseg), q_sigma, marker="o", lw=1.8, label="sigma_q")
        ax3.plot(np.arange(nseg), phi_sigma, marker="s", lw=1.6, alpha=0.85, label="sigma_phi (deg)")
        ax3.set_title("Widths vs segment")
        ax3.set_xlabel("segment index")
        ax3.set_ylabel("width")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="best", fontsize=9)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(np.arange(nseg), q_skew_peak, marker="o", lw=1.8, label="q tail asym (peak-anchored)")
        ax4.plot(np.arange(nseg), phi_skew_peak, marker="s", lw=1.6, alpha=0.85, label="phi tail asym (peak-anchored)")
        ax4.set_title("Peak-anchored tail asymmetry vs segment")
        ax4.set_xlabel("segment index")
        ax4.set_ylabel("A = (L - R) / (L + R)")
        ax4.axhline(0.0, lw=1.0, alpha=0.4)
        ax4.grid(True, alpha=0.25)
        ax4.legend(loc="best", fontsize=9)

        # (E) pattern similarity stability
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(np.arange(nseg), cos_sim, marker="o", lw=1.8, label="cosine similarity")
        ax5.plot(np.arange(nseg), pearson_r, marker="s", lw=1.6, alpha=0.85, label="pearson r")
        ax5.set_title("Pattern stability vs mean (ROI)")
        ax5.set_xlabel("segment index")
        ax5.set_ylabel("similarity")
        ax5.set_ylim(-1.05, 1.05)
        ax5.grid(True, alpha=0.25)
        ax5.legend(loc="best", fontsize=9)

        fig.suptitle(
            "Integrated intensity peak stability (10 segments)\n"
            f"ROI: q=[{q_lo:.3f},{q_hi:.3f}] Å^-1, φ=[{ph_lo:.1f},{ph_hi:.1f}] deg",
            y=0.98,
            fontsize=12,
        )
        plt.show()

    return out

def fill_map_no_nans_nearest_valid(m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Fill invalid pixels in m with the value from the nearest valid pixel (in x,y).
    Returns a dense map with no NaNs.

    m     : (H,W) float map, values only meaningful where valid==True
    valid : (H,W) bool mask of where m is valid

    Note: This extrapolates into detector regions where q/phi are undefined by the qmap.
    """
    m = np.asarray(m, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    if m.shape != valid.shape:
        raise ValueError(f"Shape mismatch: m={m.shape}, valid={valid.shape}")

    if not np.any(valid):
        raise RuntimeError("No valid pixels to extrapolate from (valid mask is empty).")

    # We fill invalid pixels using nearest-neighbor in pixel space.
    # SciPy is the simplest reliable way:
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as e:
        raise RuntimeError(
            "SciPy is required for nearest-valid filling (scipy.ndimage.distance_transform_edt not found)."
        ) from e

    # distance_transform_edt expects True for "background" to compute distances to False,
    # so we invert: invalid pixels are True background, valid pixels are False features.
    invalid = ~valid
    _, (iy_near, ix_near) = distance_transform_edt(invalid, return_indices=True)

    filled = m.copy()
    filled[invalid] = m[iy_near[invalid], ix_near[invalid]]

    # guarantee no NaNs remain
    if np.isnan(filled).any():
        raise RuntimeError("Filling failed: NaNs remain after nearest-valid extrapolation.")

    return filled

def build_q_phi_maps_from_geometry(
    f,
    *,
    shape: tuple[int, int] | None = None,
    phi_offset_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build full-detector per-pixel q and phi maps from geometry.

    Returns
    -------
    Q_map : (H,W) float64
        q magnitude in Angstrom^-1
    Phi_map_deg : (H,W) float64
        azimuth angle in degrees, from atan2(dy, dx), with dy positive DOWN
        (i.e. array row index increases downward).
        Range (-180, 180], then shifted by phi_offset_deg.
    """
    # --------- infer shape (H,W)
    if shape is None:
        if "xpcs/temporal_mean/scattering_2d" in f:
            img = f["xpcs/temporal_mean/scattering_2d"][...]
            if img.ndim == 3:
                shape = (int(img.shape[1]), int(img.shape[2]))
            else:
                shape = (int(img.shape[0]), int(img.shape[1]))
        elif "xpcs/qmap/static_roi_map" in f:
            rm = f["xpcs/qmap/static_roi_map"]
            shape = (int(rm.shape[0]), int(rm.shape[1]))
        else:
            raise ValueError("Provide shape=(H,W) or ensure scattering_2d or static_roi_map exists.")
    H, W = shape

    # --------- helpers to read scalars robustly
    def _read_first_existing(paths: list[str]) -> float:
        for p in paths:
            if p in f:
                v = f[p][...]
                # v might be scalar array
                return float(np.asarray(v).reshape(-1)[0])
        raise KeyError(f"None of these paths exist: {paths}")

    # beam center (pixels)
    # Prefer xpcs/qmap; fall back to entry/instrument if you ever pass the raw/metadata handle instead
    cx = _read_first_existing([
        "xpcs/qmap/beam_center_x",
        "entry/instrument/detector_1/beam_center_x",
        "entry/instrument/detector_1/beam_center_position_x",
    ])
    cy = _read_first_existing([
        "xpcs/qmap/beam_center_y",
        "entry/instrument/detector_1/beam_center_y",
        "entry/instrument/detector_1/beam_center_position_y",
    ])

    # detector distance (meters)
    dist_m = _read_first_existing([
        "xpcs/qmap/detector_distance",
        "entry/instrument/detector_1/distance",
    ])

    # pixel size (meters) – assume square if only one is provided
    # Your results.hdf shows xpcs/qmap/pixel_size exists.
    pix_m = _read_first_existing([
        "xpcs/qmap/pixel_size",
        "entry/instrument/detector_1/x_pixel_size",
    ])
    # If you ever want anisotropic pixels, we can extend to pix_x/pix_y.

    # energy (keV)
    E_keV = _read_first_existing([
        "xpcs/qmap/energy",
        "entry/instrument/incident_beam/incident_energy",
        "entry/instrument/monochromator/energy",
    ])

    # --------- physics: q = (4*pi/lambda) * sin(theta), where theta is half scattering angle
    # lambda [Angstrom] = 12.3984193 / E_keV
    lam_A = 12.3984193 / float(E_keV)
    k_Ainv = 2.0 * np.pi / lam_A

    # detector pixel grid
    # x increases to the right, y increases downward (array index convention)
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]

    dx_px = xx - float(cx)
    dy_px = yy - float(cy)

    # radial distance on detector face (meters)
    r_m = np.sqrt(dx_px * dx_px + dy_px * dy_px) * float(pix_m)

    # scattering angle: 2theta = arctan(r / dist)
    two_theta = np.arctan2(r_m, float(dist_m))
    theta = 0.5 * two_theta

    Q_map = 2.0 * k_Ainv * np.sin(theta)  # Å^-1

    # phi in degrees
    Phi_map_deg = np.degrees(np.arctan2(dy_px, dx_px))  # (-180, 180]
    if phi_offset_deg:
        Phi_map_deg = Phi_map_deg + float(phi_offset_deg)
        # keep it tidy
        Phi_map_deg = (Phi_map_deg + 180.0) % 360.0 - 180.0

    return Q_map.astype(np.float64), Phi_map_deg.astype(np.float64)

def exec_integrated_intensities_plot():

    integrated_intensities_plot(
        h5_file=h5_file,

        # --- data layout ---
        phi_fast_axis=True,  # True if flat index = iq*nphi + iphi
        # Set False if you ever discover iphi*nq + iq

        # --- display scaling ---
        map_scale="log",  # "log" or "linear"
        # Log is almost always what you want for scattering

        # --- robust color scaling for maps ---
        vmin_pct=1.0,  # lower percentile for color scaling
        vmax_pct=99.8,  # upper percentile (prevents Bragg peak blowout)

        # --- variability map scaling ---
        relstd_vmax=0.5,  # cap relative std map (None = auto 99.5%)
    )

    res = integrated_intensities_peak_stability(
        h5_file,
        q_range=(1.09, 1.13),
        phi_range_deg=(-10.0, 10.0),
        weight_power=1.0,  # keep tails physically important
        use_log_for_similarity=True,  # good dynamic range
        show_plots=True,
    )

    print("q drift:", res["drift_q"])
    print("phi drift (deg):", res["drift_phi_deg"])
    print("phi skew per segment:", res["skew_phi"])

def compare_q_skew_two_methods(
    I_mean_qphi: np.ndarray,
    I_seg_qphi: np.ndarray | None,
    q: np.ndarray,
    phi: np.ndarray,
    *,
    iphi: int | None = None,
    choose_phi: str = "argmax",  # "argmax" or "closest0"
    eps: float = 1e-12,
) -> dict:
    """
    Compare two q-skewness definitions:
      A) single-phi lineout:      w(q) = I(q, phi_fixed)
      B) phi-averaged lineout:    w(q) = mean_phi I(q, phi)

    Uses intensity-weighted central moments:
      q_mean = sum(w*q)/sum(w)
      mu2 = sum(w*(q-q_mean)^2)/sum(w)
      mu3 = sum(w*(q-q_mean)^3)/sum(w)
      skew = mu3 / (mu2^(3/2) + eps)

    Parameters
    ----------
    I_mean_qphi : (nq, nphi)
    I_seg_qphi  : (nseg, nq, nphi) or None
    q           : (nq,)
    phi         : (nphi,) in degrees
    iphi        : optional explicit phi index. If None, it is chosen by choose_phi.
    choose_phi  : if iphi is None:
                    - "argmax": pick phi index at global argmax intensity in mean map
                    - "closest0": pick phi index closest to 0 degrees
    """

    def _weighted_skew_1d(x: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float]:
        x = np.asarray(x, dtype=np.float64).ravel()
        w = np.asarray(w, dtype=np.float64).ravel()

        w = np.clip(w, 0.0, None)
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= eps:
            return float("nan"), float("nan"), float("nan"), float("nan")

        mu = float(np.sum(w * x) / sw)
        dx = x - mu
        mu2 = float(np.sum(w * dx * dx) / sw)
        mu3 = float(np.sum(w * dx * dx * dx) / sw)
        sig = float(np.sqrt(max(mu2, 0.0)))
        skew = float(mu3 / (mu2 ** 1.5 + eps))
        return mu, sig, skew, sw

    I_mean_qphi = np.asarray(I_mean_qphi, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()

    nq, nphi = I_mean_qphi.shape
    if q.size != nq or phi.size != nphi:
        raise ValueError(f"Shape mismatch: I_mean_qphi={I_mean_qphi.shape}, q={q.shape}, phi={phi.shape}")

    # ---- choose iphi if not provided ----
    if iphi is None:
        choose_phi = str(choose_phi).strip().lower()
        if choose_phi == "closest0":
            iphi = int(np.argmin(np.abs(phi - 0.0)))
        elif choose_phi == "argmax":
            flat = int(np.nanargmax(I_mean_qphi))
            _, iphi = np.unravel_index(flat, I_mean_qphi.shape)
        else:
            raise ValueError("choose_phi must be 'argmax' or 'closest0'")

    if not (0 <= int(iphi) < nphi):
        raise ValueError(f"iphi out of range: iphi={iphi}, nphi={nphi}")

    # ---- Method A: single-phi slice ----
    wA = I_mean_qphi[:, int(iphi)]
    q_mean_A, sigma_q_A, skew_q_A, swA = _weighted_skew_1d(q, wA)

    # ---- Method B: phi-averaged ----
    wB = np.nanmean(I_mean_qphi, axis=1)
    q_mean_B, sigma_q_B, skew_q_B, swB = _weighted_skew_1d(q, wB)

    print("q-skew comparison (mean map):")
    print(f"  phi index used: iphi={int(iphi)}  phi={phi[int(iphi)]:.3f} deg")
    print(f"  Method A (single-phi lineout):   q_mean={q_mean_A:.6f}  sigma_q={sigma_q_A:.6f}  skew_q={skew_q_A:.6f}")
    print(f"  Method B (phi-averaged lineout): q_mean={q_mean_B:.6f}  sigma_q={sigma_q_B:.6f}  skew_q={skew_q_B:.6f}")

    out = {
        "iphi_used": int(iphi),
        "phi_deg_used": float(phi[int(iphi)]),
        "mean_map": {
            "method_A_single_phi": {"q_mean": q_mean_A, "sigma_q": sigma_q_A, "skew_q": skew_q_A, "sum_w": swA},
            "method_B_phi_avg":    {"q_mean": q_mean_B, "sigma_q": sigma_q_B, "skew_q": skew_q_B, "sum_w": swB},
        },
    }

    # ---- per-segment comparison (optional) ----
    if I_seg_qphi is not None:
        I_seg_qphi = np.asarray(I_seg_qphi, dtype=np.float64)
        if I_seg_qphi.ndim != 3 or I_seg_qphi.shape[1:] != (nq, nphi):
            raise ValueError(f"Expected I_seg_qphi shape (nseg,{nq},{nphi}), got {I_seg_qphi.shape}")

        nseg = I_seg_qphi.shape[0]
        seg_rows = []
        for s in range(nseg):
            wA_s = I_seg_qphi[s, :, int(iphi)]
            wB_s = np.nanmean(I_seg_qphi[s, :, :], axis=1)

            q_mean_A_s, sigma_A_s, skew_A_s, _ = _weighted_skew_1d(q, wA_s)
            q_mean_B_s, sigma_B_s, skew_B_s, _ = _weighted_skew_1d(q, wB_s)

            seg_rows.append((s, q_mean_A_s, skew_A_s, q_mean_B_s, skew_B_s))

        print("\nq-skew comparison per segment:")
        print("  seg   q_mean(A)     skew(A)      q_mean(B)     skew(B)")
        for s, qmA, skA, qmB, skB in seg_rows:
            print(f"  {s:>3d}  {qmA:>10.6f}  {skA:>10.6f}   {qmB:>10.6f}  {skB:>10.6f}")

        out["per_segment"] = [
            {"seg": int(s), "q_mean_A": float(qmA), "skew_A": float(skA), "q_mean_B": float(qmB), "skew_B": float(skB)}
            for (s, qmA, skA, qmB, skB) in seg_rows
        ]

    return out

def _peak_anchor_from_map(I_mean_qphi: np.ndarray, q: np.ndarray, phi: np.ndarray) -> dict:
    """
    Find argmax anchor on the (nq, nphi) mean map.
    Returns iq0, iphi0, q0, phi0_deg, I0.
    """
    if I_mean_qphi.ndim != 2:
        raise ValueError(f"I_mean_qphi must be 2D (nq,nphi), got {I_mean_qphi.shape}")
    nq, nphi = I_mean_qphi.shape
    if q.size != nq or phi.size != nphi:
        raise ValueError(f"Axis mismatch: I_mean_qphi={I_mean_qphi.shape}, q={q.size}, phi={phi.size}")

    flat = int(np.nanargmax(I_mean_qphi))
    iq0, iphi0 = np.unravel_index(flat, I_mean_qphi.shape)
    q0 = float(q[iq0])
    phi0 = float(phi[iphi0])  # degrees
    I0 = float(I_mean_qphi[iq0, iphi0])
    return {"iq0": int(iq0), "iphi0": int(iphi0), "q0": q0, "phi0_deg": phi0, "I0": I0}


def _tail_asymmetry_about_q0(
    Iq: np.ndarray,
    q: np.ndarray,
    q0: float,
    *,
    p: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """
    Peak-anchored tail asymmetry A(p) about q0.
      A < 0 means heavier tail to LOWER q.
      A > 0 means heavier tail to HIGHER q.

    Uses weights w = max(Iq, 0).
    """
    Iq = np.asarray(Iq, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if Iq.size != q.size:
        raise ValueError(f"Iq and q must have same length, got {Iq.size} and {q.size}")

    w = np.clip(Iq, 0.0, None)

    dq = q - float(q0)
    left = dq < 0
    right = dq > 0

    if not np.any(left) or not np.any(right):
        return float("nan")

    dl = np.abs(dq[left]) ** float(p)
    dr = np.abs(dq[right]) ** float(p)

    ML = float(np.sum(w[left] * dl))
    MR = float(np.sum(w[right] * dr))

    denom = ML + MR
    if denom <= eps or not np.isfinite(denom):
        return float("nan")

    return float((MR - ML) / denom)


def _skew_peak_about_q0(
    Iq: np.ndarray,
    q: np.ndarray,
    q0: float,
    *,
    eps: float = 1e-12,
) -> float:
    """
    Peak-anchored "moment skewness" about q0:
      skew_peak = sum(w*dq^3) / (sum(w*dq^2))^(3/2)
    Uses weights w = max(Iq, 0).
    """
    Iq = np.asarray(Iq, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if Iq.size != q.size:
        raise ValueError(f"Iq and q must have same length, got {Iq.size} and {q.size}")

    w = np.clip(Iq, 0.0, None)
    dq = q - float(q0)

    mu2 = float(np.sum(w * dq * dq))
    mu3 = float(np.sum(w * dq * dq * dq))

    if not np.isfinite(mu2) or mu2 <= eps:
        return float("nan")

    return float(mu3 / (mu2 ** 1.5 + eps))


def compare_peak_anchored_q_tail_metrics(
    I_mean_qphi: np.ndarray,
    I_seg_qphi: np.ndarray,
    q: np.ndarray,
    phi_deg: np.ndarray,
    *,
    p_list: tuple[float, ...] = (0.0, 1.0),
    include_skew_peak: bool = True,
) -> dict:
    """
    Compare peak-anchored q-tail metrics for:
      Method A: single-phi lineout at the argmax phi (iphi0)
      Method B: phi-averaged lineout

    Inputs
    ------
    I_mean_qphi : (nq, nphi)
    I_seg_qphi  : (nseg, nq, nphi)
    q           : (nq,)
    phi_deg     : (nphi,)  degrees

    Prints:
      - anchor info (iq0, iphi0, q0, phi0)
      - mean-map A(p) and (optional) skew_peak for A and B
      - per-segment table for the same quantities

    Returns
    -------
    dict with anchor + arrays of metrics.
    """
    I_mean_qphi = np.asarray(I_mean_qphi, dtype=np.float64)
    I_seg_qphi = np.asarray(I_seg_qphi, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    phi_deg = np.asarray(phi_deg, dtype=np.float64)

    if I_mean_qphi.ndim != 2:
        raise ValueError(f"I_mean_qphi must be (nq,nphi), got {I_mean_qphi.shape}")
    if I_seg_qphi.ndim != 3:
        raise ValueError(f"I_seg_qphi must be (nseg,nq,nphi), got {I_seg_qphi.shape}")

    nq, nphi = I_mean_qphi.shape
    if q.size != nq or phi_deg.size != nphi:
        raise ValueError(
            f"Axis mismatch: I_mean_qphi={I_mean_qphi.shape}, I_seg_qphi={I_seg_qphi.shape}, "
            f"q={q.size}, phi={phi_deg.size}"
        )
    if I_seg_qphi.shape[1:] != (nq, nphi):
        raise ValueError(f"I_seg_qphi second/third dims must match mean map, got {I_seg_qphi.shape}")

    anchor = _peak_anchor_from_map(I_mean_qphi, q, phi_deg)
    iq0 = anchor["iq0"]
    iphi0 = anchor["iphi0"]
    q0 = anchor["q0"]
    phi0 = anchor["phi0_deg"]

    print("Peak anchor from mean map (argmax):")
    print(f"  iq0={iq0}, iphi0={iphi0}, q0={q0:.6f}, phi0={phi0:.3f} deg, I0={anchor['I0']:.6g}")

    # ---- mean-map lineouts ----
    Iq_A_mean = I_mean_qphi[:, iphi0]             # Method A
    Iq_B_mean = I_mean_qphi.mean(axis=1)          # Method B

    def _metrics_for_lineout(Iq: np.ndarray) -> dict:
        out = {}
        for p in p_list:
            out[f"A_p{p:g}"] = _tail_asymmetry_about_q0(Iq, q, q0, p=p)
        if include_skew_peak:
            out["skew_peak"] = _skew_peak_about_q0(Iq, q, q0)
        return out

    mA = _metrics_for_lineout(Iq_A_mean)
    mB = _metrics_for_lineout(Iq_B_mean)

    print("\nPeak-anchored q-tail metrics (mean map):")
    print(f"  Method A: single-phi lineout at iphi0={iphi0} (phi={phi0:.3f} deg)")
    for p in p_list:
        print(f"    A(p={p:g}) = {mA[f'A_p{p:g}']:+.6f}")
    if include_skew_peak:
        print(f"    skew_peak  = {mA['skew_peak']:+.6f}")

    print("  Method B: phi-averaged lineout")
    for p in p_list:
        print(f"    A(p={p:g}) = {mB[f'A_p{p:g}']:+.6f}")
    if include_skew_peak:
        print(f"    skew_peak  = {mB['skew_peak']:+.6f}")

    # ---- per-segment metrics ----
    nseg = int(I_seg_qphi.shape[0])

    A_A = {p: np.full((nseg,), np.nan, dtype=np.float64) for p in p_list}
    A_B = {p: np.full((nseg,), np.nan, dtype=np.float64) for p in p_list}
    skewA = np.full((nseg,), np.nan, dtype=np.float64) if include_skew_peak else None
    skewB = np.full((nseg,), np.nan, dtype=np.float64) if include_skew_peak else None

    for s in range(nseg):
        Iseg = I_seg_qphi[s]
        Iq_A = Iseg[:, iphi0]
        Iq_B = Iseg.mean(axis=1)

        for p in p_list:
            A_A[p][s] = _tail_asymmetry_about_q0(Iq_A, q, q0, p=p)
            A_B[p][s] = _tail_asymmetry_about_q0(Iq_B, q, q0, p=p)

        if include_skew_peak:
            skewA[s] = _skew_peak_about_q0(Iq_A, q, q0)
            skewB[s] = _skew_peak_about_q0(Iq_B, q, q0)

    print("\nPer-segment peak-anchored q-tail metrics:")
    header = "  seg"
    for p in p_list:
        header += f"   A(p={p:g})_A    A(p={p:g})_B"
    if include_skew_peak:
        header += "    skewA      skewB"
    print(header)

    for s in range(nseg):
        row = f"  {s:>3d}"
        for p in p_list:
            row += f"   {A_A[p][s]:+10.6f}  {A_B[p][s]:+10.6f}"
        if include_skew_peak:
            row += f"   {skewA[s]:+9.6f}  {skewB[s]:+9.6f}"
        print(row)

    return {
        "anchor": anchor,
        "p_list": tuple(float(p) for p in p_list),
        "mean": {"methodA": mA, "methodB": mB},
        "per_segment": {
            "A_methodA": {float(p): A_A[p] for p in p_list},
            "A_methodB": {float(p): A_B[p] for p in p_list},
            "skew_methodA": skewA,
            "skew_methodB": skewB,
        },
    }

def compare_peak_anchored_phi_tail_metrics(
    I_mean_qphi,
    I_seg_qphi,
    q,
    phi_deg,
    *,
    p_list=(0.0, 1.0),
    include_skew_peak=True,
):
    """
    Peak-anchored phi-tail asymmetry, fully analogous to q Method A.

    Uses a single-q lineout at iq0 (argmax in mean map).
    """

    Iq = I_mean_qphi.mean(axis=1)
    iq0 = int(np.argmax(Iq))

    # Mean-map lineout at the peak q
    I_phi = I_mean_qphi[iq0, :]
    iphi0 = int(np.argmax(I_phi))

    phi0 = float(phi_deg[iphi0])

    # Signed angular distance from peak (wrapped to [-180,180])
    dphi = (phi_deg - phi0 + 180.0) % 360.0 - 180.0

    def tail_asymmetry(x, I, p):
        left = I[x < 0]
        right = I[x > 0]
        xl = np.abs(x[x < 0])
        xr = np.abs(x[x > 0])

        Il = np.sum(left * xl**p)
        Ir = np.sum(right * xr**p)

        denom = Il + Ir + 1e-15
        return (Ir - Il) / denom

    out = {
        "iphi0": iphi0,
        "phi0_deg": phi0,
        "skew_peak": {},
        "per_segment": {},
    }

    # Mean map
    for p in p_list:
        out["skew_peak"][p] = tail_asymmetry(dphi, I_phi, p)

    # Per segment
    nseg = I_seg_qphi.shape[0]
    for s in range(nseg):
        Iphi_s = I_seg_qphi[s, iq0, :]
        out["per_segment"][s] = {
            p: tail_asymmetry(dphi, Iphi_s, p) for p in p_list
        }

    return out

def scroll_segments_ax0(*, log_eps: float = 1e-12, cmap: str = "magma"):
    with h5py.File(h5_file, "r") as f:
        Iseg = np.asarray(f["xpcs/temporal_mean/scattering_1d_segments"][...], dtype=np.float64)
        q = np.asarray(f["xpcs/qmap/static_v_list_dim0"][...], dtype=np.float64)
        phi = np.asarray(f["xpcs/qmap/static_v_list_dim1"][...], dtype=np.float64)

    # Sanity, reshape with phi as fast axis: flat index = iq*nphi + iphi
    nseg = int(Iseg.shape[0])
    nq = int(q.size)
    nphi = int(phi.size)
    if Iseg.ndim != 2 or Iseg.shape[1] != nq * nphi:
        raise ValueError(f"Expected Iseg shape (nseg, {nq*nphi}), got {Iseg.shape}")

    I_seg_qphi = Iseg.reshape(nseg, nq, nphi)  # (seg, q, phi)

    # For stable color scaling across segments
    I_all_pos = I_seg_qphi[I_seg_qphi > 0]
    if I_all_pos.size == 0:
        raise RuntimeError("No positive intensities found.")
    vmin = np.percentile(np.log10(I_all_pos + log_eps), 1.0)
    vmax = np.percentile(np.log10(I_all_pos + log_eps), 99.7)

    state = {"s": 0}

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.8))

    def seg_to_image(s: int):
        # ax0-style: x=q, y=phi, so image must be (phi, q)
        Iphiq = I_seg_qphi[s].T  # (phi, q)
        return np.log10(np.clip(Iphiq, 0.0, None) + log_eps)

    im = ax.imshow(
        seg_to_image(0),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[q.min(), q.max(), phi.min(), phi.max()],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("q (Å$^{-1}$)")
    ax.set_ylabel("φ (deg)")
    title = ax.set_title(f"scattering_1d_segments, log10(I+eps), segment 0/{nseg-1}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(I + eps)")

    def redraw():
        s = state["s"]
        im.set_data(seg_to_image(s))
        title.set_text(f"scattering_1d_segments, log10(I+eps), segment {s}/{nseg-1}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "d"):
            state["s"] = (state["s"] + 1) % nseg
            redraw()
        elif event.key in ("left", "a"):
            state["s"] = (state["s"] - 1) % nseg
            redraw()
        elif event.key in ("home",):
            state["s"] = 0
            redraw()
        elif event.key in ("end",):
            state["s"] = nseg - 1
            redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()


file_ID = 'A073'

if file_ID == 'A013':
    filename = (r'/Users/emilioescauriza/Desktop/A013_IPA_NBH_1_att0100_079K_001_results.hdf')
elif file_ID == 'A073':
    filename = (r'/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_IPA_NBH_1_att0100_260K_001_results.hdf')
else:
    base = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")
    filename = next(base.glob(f"{file_ID}_*_results.hdf"))
h5_file = filename


if __name__ == "__main__":

    # h5_file_inspector(h5_file)
    # g2_plotter(h5_file)
    # ttc_plotter(h5_file)
    # intensity_vs_time(h5_file)
    # static_vs_dynamic_bins(h5_file)
    # combined_plot(h5_file)
    # oauth_test()
    # image_upload()
    # figure_upload()
    # q_spacing_inspector(h5_file)
    # integrated_intensities_inspector(h5_file)
    # execute_find_bragg_peak_center_from_scattering_2d_with_overlay()
    # exec_make_and_save_inferred_qphi_maps()
    # exec_quick_check_inferred_qphi_npz()
    # exec_bragg_peak_shape_metrics_fixed_q_phi()
    # exec_build_q_phi_map()
    exec_integrated_intensities_plot()
    # scroll_segments_ax0()





    pass