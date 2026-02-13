import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib as mpl
mpl.use('macosx')
from scipy.signal import correlate2d
import h5py
import hdf5plugin
import sys
import matplotlib.pyplot as plt
from xpcs import *
from sims import *
# from autocorrelations import *
import cv2
from scipy.special import erfinv
import pyopencl as cl
from pathlib import Path
import json
from scipy.fft import fft, ifft, fftfreq
import gspread
from io import BytesIO
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload




def find_results_hdf(base_dir: Path, sample_id: str) -> Path | None:
    pattern = f"{sample_id}_*_results.hdf"
    matches = sorted(base_dir.glob(pattern))
    return matches[0] if matches else None

def get_ids_for_position(ws, position_name, id_col=1, position_col=3, header_rows=1):
    """
    Return a list of (row_number, sample_id) for rows where column C == position_name.

    Parameters
    ----------
    ws : gspread.Worksheet
    position_name : str
        e.g. "A5"
    id_col : int
        Column index for sample ID (A = 1)
    position_col : int
        Column index for position name (C = 3)
    header_rows : int
        Number of header rows to skip (usually 1)

    Returns
    -------
    list of (int, str)
        [(row_number, sample_id), ...]
    """
    col_ids = ws.col_values(id_col)
    col_pos = ws.col_values(position_col)

    n = min(len(col_ids), len(col_pos))
    results = []

    for i in range(header_rows, n):
        if col_pos[i].strip() == position_name:
            results.append((i + 1, col_ids[i].strip()))

    return results

def find_rows_with_position(ws, position_name="A5"):
    """
    Returns a list of row numbers where column C equals position_name.
    Assumes row 1 is header.
    """
    col_a = ws.col_values(1)  # Column A = index 1
    col_c = ws.col_values(3)  # Column C = index 3

    matching_rows = [
        i + 1                      # convert 0-based index → 1-based row
        for i, val in enumerate(col_c)
        if val == position_name
    ]

    ids = [
        col_a[i]
        for i in range(1, min(len(col_a), len(col_c)))
        if col_c[i] == position_name
    ]

    return ids, matching_rows

def get_rows_and_ids_for_position(ws, position_name="A5", id_col=1, position_col=3, header_rows=1):
    col_a = ws.col_values(id_col)
    col_c = ws.col_values(position_col)

    start = header_rows  # 0-based index into lists
    n = min(len(col_a), len(col_c))

    out = []
    for i in range(start, n):
        if col_c[i].strip() == position_name:
            out.append((i + 1, col_a[i].strip()))  # (sheet_row_number, ID)
    return out


def find_results_hdf(base_dir: Path, sample_id: str) -> Path | None:
    pattern = f"{sample_id}_*_results.hdf"
    matches = sorted(base_dir.glob(pattern))
    return matches[0] if matches else None

def rows_to_cells(rows, column_letter="AF"):
    return [f"{column_letter}{r}" for r in rows]


def display_images(image_array, coords):
    fig, ax = plt.subplots()
    total_images = image_array.shape[0]
    current_index = 0

    # Display the first image
    new_cmap = new_colour_map()
    im = ax.imshow(image_array[current_index], cmap=new_cmap)
    cbar = fig.colorbar(im, ax=ax)  # Store colorbar reference
    cbar.set_label('ADU')
    title = ax.set_title(f"Index: {coords[current_index]}")

    def onkey(event):
        nonlocal current_index, cbar  # Keep track of colorbar

        if event.key == "right":  # Right arrow key
            current_index = (current_index + 1) % total_images
        elif event.key == "left":  # Left arrow key
            current_index = (current_index - 1) % total_images

        # Update image data
        im.set_array(image_array[current_index])

        # Update color limits dynamically
        im.set_clim(vmin=image_array[current_index].min(), vmax=image_array[current_index].max())

        # Remove previous colorbar
        if cbar:
            cbar.remove()

        # Add new colorbar to reflect updated intensity range
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('ADU')

        # Update title
        title.set_text(f"Index: {coords[current_index]}")

        # Redraw figure
        fig.canvas.draw()

    # Connect the keyboard event to the function
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

def generate_random_speckle_image(shape=(512, 1024), mean_intensity=1.0, contrast=1.0, seed=None):
    """
    Generate a 2D speckle pattern with randomly distributed intensities.

    Args:
        shape: Tuple defining the shape of the image (height, width).
        mean_intensity: Mean value of the speckle intensity.
        contrast: Degree of contrast. 1.0 means fully developed speckle.
        seed: Optional random seed for reproducibility.

    Returns:
        speckle_image: 2D numpy array of speckle intensities.
    """
    if seed is not None:
        np.random.seed(seed)

    # Fully developed speckle follows a negative exponential distribution
    scale = mean_intensity / contrast
    speckle_image = np.random.exponential(scale=scale, size=shape)

    return speckle_image

def create_2D_guassian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Defines a 2D Gaussian function.

    Args:
        xy: Tuple of flattened x and y coordinate arrays (x.ravel(), y.ravel()).
        amplitude: Peak amplitude of the Gaussian.
        xo: Center x-coordinate of the Gaussian.
        yo: Center y-coordinate of the Gaussian.
        sigma_x: Standard deviation in the x-direction.
        sigma_y: Standard deviation in the y-direction.
        theta: Rotation angle of the Gaussian in radians.
        offset: Constant offset of the Gaussian.

    Returns:
        Flattened 2D Gaussian values corresponding to the input coordinates.
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    g = offset + amplitude * np.exp(-(a * (x - xo)**2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo)**2))
    return g.ravel()

def generate_speckle_sequence(
    shape,
    num_frames,
    similarity,  # 1.0 = identical, 0.0 = fully random
    seed
    ):

    rng = np.random.default_rng(seed)
    base_phase = rng.uniform(0, 2 * np.pi, size=shape)
    amplitude = np.ones(shape)  # Uniform illumination

    sequence = []
    phase = base_phase.copy()
    decay_factor = similarity  # Controls exponential decay

    for i in range(num_frames):
        # Exponentially decaying correlation
        noise = rng.normal(0, 1, size=shape)
        phase = decay_factor * phase + np.sqrt(1 - decay_factor**2) * noise
        current_phase = phase % (2 * np.pi)
        field = amplitude * np.exp(1j * current_phase)
        field -= np.mean(field)  # Remove DC component
        speckle = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2
        sequence.append(speckle / np.mean(speckle))  # Normalize

    sequence = np.array(sequence)

    return sequence

def generate_speckle_sequence_with_exponential_decay(
    shape,
    num_frames,
    similarity,  # 1.0 = identical, 0.0 = fully random
    seed
    ):

    rng = np.random.default_rng(seed)
    base_phase = rng.uniform(0, 2 * np.pi, size=shape)
    amplitude = np.ones(shape)  # Uniform illumination

    sequence = []
    phase = base_phase.copy()
    decay_factor = similarity  # Controls exponential decay

    tau_0 = 10  # Time constant for exponential decay


    for i in range(num_frames):
        # Exponentially decaying correlation
        if i == 0:
            decay_factor = 1
        else:
            decay_factor = np.exp(-(i + 1) / tau_0) / np.exp(-i / tau_0)
        print(decay_factor)
        noise = rng.normal(0, 1, size=shape)
        phase = decay_factor * phase + np.sqrt(1 - decay_factor**2) * noise
        current_phase = phase % (2 * np.pi)
        field = amplitude * np.exp(1j * current_phase)
        field -= np.mean(field)  # Remove DC component
        speckle = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2
        sequence.append(speckle / np.mean(speckle))  # Normalize

    sequence = np.array(sequence)

    return sequence

def new_colour_map():
    base_cmap = get_cmap('viridis')  # diverging colormap
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[0] = [0, 0, 0, 1]
    new_cmap = ListedColormap(colors)
    return new_cmap

def display_images(image_array, coords):
    fig, ax = plt.subplots()
    total_images = image_array.shape[0]
    current_index = 0

    # Display the first image
    new_cmap = new_colour_map()
    im = ax.imshow(image_array[current_index], cmap=new_cmap)
    cbar = fig.colorbar(im, ax=ax)  # Store colorbar reference
    cbar.set_label('ADU')
    title = ax.set_title(f"Index: {coords[current_index]}")

    def onkey(event):
        nonlocal current_index, cbar  # Keep track of colorbar

        if event.key == "right":  # Right arrow key
            current_index = (current_index + 1) % total_images
        elif event.key == "left":  # Left arrow key
            current_index = (current_index - 1) % total_images

        # Update image data
        im.set_array(image_array[current_index])

        # Update color limits dynamically
        im.set_clim(vmin=image_array[current_index].min(), vmax=image_array[current_index].max())

        # Remove previous colorbar
        if cbar:
            cbar.remove()

        # Add new colorbar to reflect updated intensity range
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('ADU')

        # Update title
        title.set_text(f"Index: {coords[current_index]}")

        # Redraw figure
        fig.canvas.draw()

    # Connect the keyboard event to the function
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

def creat_synthetic_xpcs_data(x_size, y_size,  # image dimensions
                              gauss_cen_x, gauss_cen_y,  # center of the Gaussian
                              gauss_sigma_x, gauss_sigma_y,  # standard deviations of the Gaussian
                              gauss_theta,  # rotation angle of the Gaussian in radians
                              gauss_offset,  # offset of the Gaussian
                              number_of_frames,  # number of frames in the sequence
                              similarity, # 1.0 = identical, 0.0 = fully random
                              seed  = 0  # random seed for reproducibility
                              ):

    # Create synthetic data with a 2D Gaussian
    x = np.arange(0, x_size)
    y = np.arange(0, y_size)
    x, y = np.meshgrid(x, y)
    gaussian_mask = create_2D_guassian((x, y), 1, gauss_cen_x, gauss_cen_y, gauss_sigma_x, gauss_sigma_y,
                                       gauss_theta, gauss_offset).reshape(y_size, x_size)
    gaussian_mask = gaussian_mask - np.min(gaussian_mask)  # Normalize to have a max of 0

    speckle_sequence = generate_speckle_sequence((y_size, x_size), number_of_frames, similarity, seed)

    bragg_peaks = speckle_sequence * gaussian_mask

    return bragg_peaks


def processed_data_loader():
    """""""""
    --- Taken from Erik's Jupyter notebooks ---
    """""""""

    ### Beamline Parameters ###
    Beamline = 'P10'
    Geometry = 'Horizontal'
    Compound = 'NPS'

    ### Analysis Parameters ###

    ### File Parameters ###
    Particle = 2  # 2,3,A7
    Temperature = 298  # [298,305,315,325,345,365,385,405,350_cool] NBH = [320,329,339,360,380,399,420]
    Scan = 219  # P2: [219,334,416,552,579,709,779,796,868], P3:[239,356 NBH = [343,344,349,353,358,362,363]
    Attenuation = 20

    # Crop Size
    # number of standard deviations from peak
    Xv = 5
    Xh = 5

    # Binning Parameters
    # Number of pixels in every direction to bin
    # A bin size of 1 does not bin
    binSize = 1

    timeBatch = -1  # seconds
    startTime = 0  # seconds

    stability = True

    # Mask Parameters
    New_Mask = True
    Tilted_Mask = False
    maskSize = 3  # standard deviations
    numRings = 10  # Number of elliptical rings
    numSlices = 1  # Number slices, should be even
    tol = 0.02  # Percentage tolerance for converging to ring partition
    res = 0.0001  # Resolution for ring size increments

    # Define the path to the configuration file
    config_path = Path("/Users/emilioescauriza/Documents/repos/002_XPCS_analysis_development/erik_file_transfer/notebooks/config.json")

    # Open and load the JSON configuration file
    with open(config_path, "r") as file:
        config = json.load(file)

    BASE_DIR = Path(config.get("NPS_base"))
    mask_path = Path(config.get(f"NPS_Mask_Result_{Xv}x{Xh}"))
    dataset_name = f'Particle_{Particle}/{Temperature}K/NPS_01_00{Scan}'
    subfolder = 'e4m'
    filename = f'{Compound}_01_00{Scan}_master.h5'
    batchname = f'{Compound}_01_00{Scan}.batchinfo'
    # Set path to working directory
    work_path = BASE_DIR / f'{Compound}_01_00{Scan}' / subfolder
    # processed_path = work_path / 'processed' / f'particle{Particle}_temp{Temperature}K_scan{Scan}_crop{Xv}x{Xh}.h5'
    processed_path = work_path / 'processed' / f'{Compound}_particle{Particle}_temp{Temperature}.0K_crop{Xv}x{Xh}_NewMask{New_Mask}_stability{stability}.h5'

    results_path = work_path / 'results'
    results = results_path / f'particle{Particle}_temp{Temperature}.0K_scan{Scan}_crop{Xv}x{Xh}_ms{maskSize}_nr{numRings}ns{numSlices}_bin{binSize}_st{startTime}_tb{int(timeBatch)}_newmask{New_Mask}_tilt{Tilted_Mask}.h5'

    # Check that path exists
    if work_path.exists():
        print(f'Directory found: {work_path}')
    else:
        print(f'Directory not found: {work_path}')

    # Load File Names
    file_list = np.sort(os.listdir(work_path)).tolist()

    # Remove first element if it's DS_Store
    if file_list[0] == '.DS_Store':
        file_list.remove(file_list[0])

    # Removes strange file that occurs if you edit batchinfo
    if file_list[0][0:2] == '._':
        file_list.remove(file_list[0])

    print('Contains:')
    for f in file_list:
        print(f)

    # Check that the processed file exists
    if not os.path.exists(processed_path):
        print(f"File '{processed_path}' does not exist! Please create it in pre_processing.")

    # Load file
    with h5py.File(processed_path, 'r') as DATA:

        pixelSize = DATA['experimental_parameters/pixel_size'][:][0]
        frameSpacing = DATA['experimental_parameters/frame_spacing'][:][0]
        nFrames = DATA['experimental_parameters/nFrames'][:][0]
        tilt = DATA['pre_processing/angular_tilt'][:][0]
        Qx = DATA['pre_processing/Qx'][:]
        Qy = DATA['pre_processing/Qy'][:]
        hSig = DATA['pre_processing/hSig'][:][0]
        vSig = DATA['pre_processing/vSig'][:][0]

        if timeBatch == -1:
            timeBatch = nFrames * frameSpacing

        POPT = DATA['gaussian_fitting/POPT'][:][
               int(startTime // frameSpacing):int(startTime // frameSpacing) + int(timeBatch // frameSpacing), :]
        PCOV = DATA['gaussian_fitting/PCOV'][:][
               int(startTime // frameSpacing):int(startTime // frameSpacing) + int(timeBatch // frameSpacing), :]

        # Take timeChunk of data. Not whole scan
        det = DATA['data/det'][:][
              int(startTime // frameSpacing):int(startTime // frameSpacing) + int(timeBatch // frameSpacing), :, :]
        det_corr = DATA['data/det_corr'][:][
                   int(startTime // frameSpacing):int(startTime // frameSpacing) + int(timeBatch // frameSpacing), :, :]

    return det, det_corr, pixelSize, frameSpacing, nFrames, tilt, Qx, Qy, hSig, vSig, POPT, PCOV

def processed_data_loader_frame_range(processed_path, start_frame=0, n_frames=None):
    """""""""
    --- Taken from Erik's Jupyter notebooks ---
    """""""""

    ### Beamline Parameters ###
    Beamline = 'P10'
    Geometry = 'Horizontal'
    Compound = 'NPS'

    ### Analysis Parameters ###

    ### File Parameters ###
    Particle = 2  # 2,3,A7
    Temperature = 298  # [298,305,315,325,345,365,385,405,350_cool] NBH = [320,329,339,360,380,399,420]
    Scan = 219  # P2: [219,334,416,552,579,709,779,796,868], P3:[239,356 NBH = [343,344,349,353,358,362,363]
    Attenuation = 20

    # Crop Size
    # number of standard deviations from peak
    Xv = 5
    Xh = 5

    # Binning Parameters
    # Number of pixels in every direction to bin
    # A bin size of 1 does not bin
    binSize = 1

    timeBatch = -1  # seconds
    startTime = 0  # seconds

    stability = True

    # Mask Parameters
    New_Mask = True
    Tilted_Mask = False
    maskSize = 3  # standard deviations
    numRings = 10  # Number of elliptical rings
    numSlices = 1  # Number slices, should be even
    tol = 0.02  # Percentage tolerance for converging to ring partition
    res = 0.0001  # Resolution for ring size increments

    # Define the path to the configuration file
    config_path = Path("/Users/emilioescauriza/Documents/repos/002_XPCS_analysis_development/erik_file_transfer/notebooks/config.json")

    # Open and load the JSON configuration file
    with open(config_path, "r") as file:
        config = json.load(file)

    BASE_DIR = Path(config.get("NPS_base"))
    mask_path = Path(config.get(f"NPS_Mask_Result_{Xv}x{Xh}"))
    dataset_name = f'Particle_{Particle}/{Temperature}K/NPS_01_00{Scan}'
    subfolder = 'e4m'
    filename = f'{Compound}_01_00{Scan}_master.h5'
    batchname = f'{Compound}_01_00{Scan}.batchinfo'
    # Set path to working directory
    work_path = BASE_DIR / dataset_name / subfolder
    # processed_path = work_path / 'processed' / f'particle{Particle}_temp{Temperature}K_scan{Scan}_crop{Xv}x{Xh}.h5'
    processed_path = work_path / 'processed' / f'particle{Particle}_temp{Temperature}K_scan{Scan}_crop{Xv}x{Xh}_NewMask{New_Mask}_stability{stability}.h5'

    results_path = work_path / 'results'
    results = results_path / f'particle{Particle}_temp{Temperature}K_scan{Scan}_crop{Xv}x{Xh}_ms{maskSize}_nr{numRings}ns{numSlices}_bin{binSize}_st{startTime}_tb{int(timeBatch)}_newmask{New_Mask}_tilt{Tilted_Mask}.h5'

    # Check that path exists
    if work_path.exists():
        print(f'Directory found: {work_path}')
    else:
        print(f'Directory not found: {work_path}')

    # Load File Names
    file_list = np.sort(os.listdir(work_path)).tolist()

    # Remove first element if it's DS_Store
    if file_list[0] == '.DS_Store':
        file_list.remove(file_list[0])

    # Removes strange file that occurs if you edit batchinfo
    if file_list[0][0:2] == '._':
        file_list.remove(file_list[0])

    print('Contains:')
    for f in file_list:
        print(f)

    # Check that the processed file exists
    if not os.path.exists(processed_path):
        print(f"File '{processed_path}' does not exist! Please create it in pre_processing.")

    # Load frames from file
    with h5py.File(processed_path, 'r') as DATA:
        pixelSize   = DATA['experimental_parameters/pixel_size'][()][0]
        frameSpacing = DATA['experimental_parameters/frame_spacing'][()][0]
        nFrames     = DATA['experimental_parameters/nFrames'][()][0]
        tilt        = DATA['pre_processing/angular_tilt'][()][0]
        Qx          = DATA['pre_processing/Qx'][:]
        Qy          = DATA['pre_processing/Qy'][:]
        hSig        = DATA['pre_processing/hSig'][()][0]
        vSig        = DATA['pre_processing/vSig'][()][0]

        # determine how many frames to load
        if n_frames is None:
            end_frame = nFrames
        else:
            end_frame = min(start_frame + n_frames, nFrames)

        # slice directly from datasets (efficient)
        POPT     = DATA['gaussian_fitting/POPT'][start_frame:end_frame, :]
        PCOV     = DATA['gaussian_fitting/PCOV'][start_frame:end_frame, :]
        det      = DATA['data/det'][start_frame:end_frame, :, :]
        det_corr = DATA['data/det_corr'][start_frame:end_frame, :, :]

    return det, det_corr, pixelSize, frameSpacing, nFrames, tilt, Qx, Qy, hSig, vSig, POPT, PCOV

def G2_loader_from_processed_data():
    (det, det_corr, pixelSize, frameSpacing, nFrames, tilt, Qx, Qy, hSig,
     vSig, POPT, PCOV) = processed_data_loader()
    # (det, det_corr, pixelSize, frameSpacing, nFrames, tilt, Qx, Qy, hSig,
    #  vSig, POPT, PCOV) = processed_data_loader_frame_range(processed_path, start_frame=start_frame, n_frames=n_frames)

    images = det_corr

    mask = np.ones(images[0].shape)

    binSize = 1
    # Mask Parameters
    New_Mask = False
    Tilted_Mask = False
    maskSize = 3  # standard deviations
    numRings = 10  # Number of elliptical rings
    numSlices = 1  # Number slices, should be even
    tol = 0.02  # Percentage tolerance for converging to ring partition
    res = 0.0001  # Resolution for ring size increments

    if New_Mask == True:
        # Create Elliptical Masks. Non-masked regions are -1
        masks, ringWidths = const_int_mask(det, hSig / binSize, vSig / binSize, maskSize, numRings,
                                           numSlices, -tilt, tol, res)  # Create the masks of equal intensity
        np.savez("masks_and_ringwidths.npz", masks=masks, ringWidths=ringWidths)
    elif New_Mask == False:
        # Load masks
        data = np.load("masks_and_ringwidths.npz")
        masks = data["masks"]
        ringWidths = data["ringWidths"]

    # Delays per level. Increase for more sampling
    dpl = 6
    m = masks[2, :, :]
    # IF are intensities future, IP intensities past. Used to normalize g2
    sumI, G2, IF, IP, GC = g2calc(images, mask, dpl)


    return G2, IF, IP, masks, frameSpacing, nFrames, dpl




