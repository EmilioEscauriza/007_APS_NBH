# bragg_forward_model.py

BCDI forward model: compute the 3D coherent diffraction intensity around a single Bragg peak via FFT.

## Purpose

Given a nanocrystal shape (support) and a uniform strain field, simulate the 3D coherent diffraction pattern that would be measured in a Bragg coherent diffraction imaging (BCDI) experiment. The model uses the kinematic approximation and is computed efficiently via FFT.

## Physics

### Kinematic BCDI forward model

The measured intensity around a Bragg peak $\mathbf{G}_{hkl}$ is:

$$I(\mathbf{Q}) = \left| \text{FT}\left[\rho(\mathbf{r}) \, e^{i \, \mathbf{G}_{hkl} \cdot \mathbf{u}(\mathbf{r})}\right] \right|^2$$

where $\mathbf{Q}$ is the deviation from the Bragg peak (i.e. $\mathbf{q} = \mathbf{G}_{hkl} + \mathbf{Q}$).

- $\rho(\mathbf{r})$: particle shape function (support), a binary 3D array.
- $\mathbf{u}(\mathbf{r})$: displacement field. For uniform strain $\boldsymbol{\varepsilon}$, $\mathbf{u}(\mathbf{r}) = \boldsymbol{\varepsilon} \cdot \mathbf{r}$.
- $\mathbf{G}_{hkl} = h\,\mathbf{b}_1 + k\,\mathbf{b}_2 + l\,\mathbf{b}_3$: Bragg vector from the reciprocal lattice.
- FT: 3D discrete Fourier transform (NumPy `fftn`).

### Phase from uniform strain

The phase at each voxel simplifies to:

$$\phi(\mathbf{r}) = \mathbf{G}_{hkl} \cdot \boldsymbol{\varepsilon} \cdot \mathbf{r}$$

Uniform strain shifts and broadens the Bragg peak without changing the fringe pattern from the particle shape.

### Reciprocal lattice

Reciprocal basis vectors $\mathbf{b}_i$ are computed from direct-space lattice parameters $(a, b, c, \alpha, \beta, \gamma)$ using the standard crystallographic convention ($\mathbf{a}_1$ along $x$, $\mathbf{a}_2$ in the $x$-$y$ plane) and the relation $\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi\,\delta_{ij}$.

### q-space resolution

$$\Delta q = \frac{2\pi}{N \cdot \Delta r}$$

The simulation covers $\pm N \Delta q / 2$ around the Bragg peak.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `energy_keV` | 12.4 | X-ray energy [keV] |
| `a`, `b`, `c` | 6.13, 6.13, 6.13 | Lattice constants [Å] (NaBH₄ cubic) |
| `alpha_deg`, `beta_deg`, `gamma_deg` | 90, 90, 90 | Lattice angles [°] |
| `hkl` | (1, 1, 0) | Miller indices of the measured reflection |
| `grid_size` | 128 | Grid dimension N (N×N×N voxels) |
| `voxel_size_nm` | 5.0 | Real-space voxel size [nm] |
| `particle_shape` | "sphere" | Support shape: "sphere", "cube", or "faceted" |
| `particle_size_nm` | 300.0 | Particle diameter/side [nm] |
| `strain_voigt` | (0,0,0,0,0,0) | Strain in Voigt notation: (xx, yy, zz, yz, xz, xy) |
| `max_photons` | None | Poisson noise scaling; None = noiseless |

### Derived quantities

| Quantity | Formula |
|----------|---------|
| `wavelength_A` | $hc / E$ |
| `d_spacing_A` | $2\pi / |\mathbf{G}_{hkl}|$ |
| `theta_B_deg` | $\arcsin(\lambda / 2d)$ |
| `delta_q_inv_A` | $2\pi / (N \cdot \Delta r)$ |

## Usage

### Static plot (default parameters)

```bash
python modelling/bragg_forward_model.py
```

### Custom reflection and strain

```bash
python modelling/bragg_forward_model.py --hkl 2 0 0 --strain 1e-3 0 0 0 0 0
```

### Interactive sliders

```bash
python modelling/bragg_forward_model.py --interactive
```

Sliders control particle size, diagonal strain components (ε_xx, ε_yy, ε_zz), and Poisson noise level. Grid size defaults to 64 in interactive mode for responsiveness.

### Save output

```bash
python modelling/bragg_forward_model.py --grid-size 256 --save output.npz
```

Saves intensity, q-axes, hkl, energy, voxel size, and strain to a compressed NumPy archive.

### All CLI options

```
--energy         X-ray energy [keV] (default 12.4)
--hkl H K L      Miller indices (default 1 1 0)
--lattice a b c alpha beta gamma   Lattice params [Å, °] (default NaBH₄ cubic)
--grid-size      N for N×N×N (default 128)
--voxel-size     Voxel size [nm] (default 5.0)
--shape          sphere | cube | faceted (default sphere)
--particle-size  Particle size [nm] (default 300)
--strain xx yy zz yz xz xy   Strain tensor, Voigt (default all zeros)
--photons        Max photons for Poisson noise (default: noiseless)
--interactive    Launch slider GUI
--save FILE      Save to .npz
```

## Defaults

The default lattice parameters correspond to NaBH₄ (sodium borohydride, cubic, $a = 6.13$ Å) at 12.4 keV (APS 08-ID-E), consistent with the rest of the repository's modelling scripts.

## Dependencies

- **numpy**, **scipy**, **matplotlib** (already used throughout the repo)
- **bcdi** (optional): only needed for `"faceted"` particle shapes via `bcdi.simulation.supportMaker`. Falls back to a sphere if bcdi is not installed.
