import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""Average axial and lateral bead resolutions (5 beads)"""

# ------------------------------------------------
# 1) File Paths
# ------------------------------------------------
folder_path = "/Users/elinestas/Desktop/Beads"

bead_files = [
    ("Stack 1 Bead 1.1 x.csv", "Stack 1 Bead 1.1 y.csv", "Stack 1 Bead 1 z.csv"),
    ("Stack 1 Bead 3 x.csv",  "Stack 1 Bead 3 y.csv",  "Stack 1 Bead 3 z.csv"),
    ("Stack 3 Bead 1 x.csv",  "Stack 3 Bead 1 y.csv",  "Stack 3 Bead 1 z.csv"),
    ("Stack 5 Bead 1 x.csv",  "Stack 5 Bead 1 y.csv",  "Stack 5 Bead 1 z.csv"),
    ("Stack 5 Bead 2 x.csv",  "Stack 5 Bead 3 y.csv",  "Stack 5 Bead 3 z.csv"),
]

FRAME_SIZE_UM = 0.2   # 1 frame = 0.2 µm
Z_RANGE       = (-6, 6)  # µm
L_RANGE       = (-6, 6)
STEP          = 0.05      # µm per point on fine grid

fine_grid_z = np.arange(Z_RANGE[0], Z_RANGE[1] + STEP/2, STEP)
fine_grid_l = np.arange(L_RANGE[0], L_RANGE[1] + STEP/2, STEP)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------
# 2) Helper Functions
# ------------------------------------------------

def read_intensity_profile(path, in_pixels=False):
    df = pd.read_csv(path)
    if in_pixels:
        return df["Frame"].to_numpy() * FRAME_SIZE_UM, df["Mean"].to_numpy()
    return df["Distance_(microns)"].to_numpy(), df["Gray_Value"].to_numpy()


def find_fwhm(x, y):
    """Full‑width at half‑maximum of y(x)."""
    if len(y) < 3 or np.all(np.isnan(y)):
        return np.nan
    peak = np.nanmax(y)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    half = peak / 2.0
    mask = y >= half
    if mask.sum() < 2:
        return np.nan
    idx = np.where(mask)[0]
    def lin_interp(i1, i2):
        return x[i1] + (x[i2]-x[i1]) * (half - y[i1]) / (y[i2]-y[i1])
    left  = lin_interp(idx[0]-1, idx[0]) if idx[0] > 0 else x[idx[0]]
    right = lin_interp(idx[-1], idx[-1]+1) if idx[-1] < len(x)-1 else x[idx[-1]]
    return right - left


def gaussian_1d(x, A, x0, sigma, offset):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

# ------------------------------------------------
# 3) Read Data & build per‑bead curves
# ------------------------------------------------
axial_curves, lateral_curves = [], []
axial_fwhm,  lateral_fwhm  = [], []

for x_name, y_name, z_name in bead_files:
    # --- axial (Z) ---
    z_dist, z_int = read_intensity_profile(os.path.join(folder_path, z_name), in_pixels=True)
    z_dist -= z_dist[np.argmax(z_int)]  # recenter
    z_norm  = z_int / z_int.max()
    z_interp = np.interp(fine_grid_z, z_dist, z_norm, left=np.nan, right=np.nan)
    axial_curves.append(z_interp)
    axial_fwhm.append(find_fwhm(fine_grid_z, z_interp))

    # --- lateral (mean of X & Y) ---
    x_dist, x_int = read_intensity_profile(os.path.join(folder_path, x_name))
    y_dist, y_int = read_intensity_profile(os.path.join(folder_path, y_name))

    x_dist -= x_dist[np.argmax(x_int)]
    y_dist -= y_dist[np.argmax(y_int)]

    ix = np.interp(fine_grid_l, x_dist, x_int, left=np.nan, right=np.nan)
    iy = np.interp(fine_grid_l, y_dist, y_int, left=np.nan, right=np.nan)
    il = np.nanmean(np.vstack([ix, iy]), axis=0)
    il /= np.nanmax(il)
    lateral_curves.append(il)
    lateral_fwhm.append(find_fwhm(fine_grid_l, il))

axial_curves   = np.array(axial_curves)
lateral_curves = np.array(lateral_curves)
axial_fwhm     = np.array(axial_fwhm,   dtype=float)
lateral_fwhm   = np.array(lateral_fwhm, dtype=float)
# ------------------------------------------------
# 4) FWHM statistics
# ------------------------------------------------
print("=== FWHM per bead ===")
for i, (lat, ax) in enumerate(zip(lateral_fwhm, axial_fwhm), 1):
    print(f"Bead {i}: lateral = {lat:.3f} µm   axial = {ax:.3f} µm")

lat_mean, ax_mean = np.nanmean(lateral_fwhm), np.nanmean(axial_fwhm)
lat_sd,   ax_sd   = np.nanstd(lateral_fwhm, ddof=1), np.nanstd(axial_fwhm, ddof=1)
N = (~np.isnan(axial_fwhm)).sum()
lat_sem, ax_sem = lat_sd / np.sqrt(N), ax_sd / np.sqrt(N)

print("\n=== Mean ± SD (± SEM) ===")
print(f"Lateral: {lat_mean:.3f} ± {lat_sd:.3f} µm  (SEM ±{lat_sem:.3f})")
print(f"Axial  : {ax_mean:.3f} ± {ax_sd:.3f} µm  (SEM ±{ax_sem:.3f})\n")

# ------------------------------------------------
# 5) Mean & SEM curves
# ------------------------------------------------
mean_ax  = np.nanmean(axial_curves,   axis=0)
mean_lat = np.nanmean(lateral_curves, axis=0)

# point‑wise SEM (handle zero valid counts)
valid_ax   = np.sum(~np.isnan(axial_curves),   axis=0)
valid_lat  = np.sum(~np.isnan(lateral_curves), axis=0)
sem_ax_all  = np.full_like(mean_ax, np.nan)
sem_lat_all = np.full_like(mean_lat, np.nan)

nonzero_ax  = valid_ax  > 1
nonzero_lat = valid_lat > 1
sem_ax_all[nonzero_ax]  = np.nanstd(axial_curves[:, nonzero_ax],   axis=0, ddof=1) / np.sqrt(valid_ax[nonzero_ax])
sem_lat_all[nonzero_lat] = np.nanstd(lateral_curves[:, nonzero_lat], axis=0, ddof=1) / np.sqrt(valid_lat[nonzero_lat])

mask_ax  = np.isfinite(mean_ax)
mask_lat = np.isfinite(mean_lat)
fg_z, ma, se_ax = fine_grid_z[mask_ax],  mean_ax[mask_ax],  sem_ax_all[mask_ax]
fg_l, ml, se_lat = fine_grid_l[mask_lat], mean_lat[mask_lat], sem_lat_all[mask_lat]

# Gaussian fits on mean curves
popt_ax  = curve_fit(gaussian_1d, fg_z, ma, p0=[1, 0, 4, 0])[0]
ax_fit   = gaussian_1d(fg_z, *popt_ax)

popt_lat = curve_fit(gaussian_1d, fg_l, ml, p0=[1, 0, 2, 0])[0]
lat_fit  = gaussian_1d(fg_l, *popt_lat)

plt.rcParams.update({"figure.facecolor": "white"})
# ------------------------------------------------
# 6) Axial plot
# ------------------------------------------------
plt.figure(figsize=(5, 5))
plt.fill_between(fg_z, ma - se_ax, ma + se_ax, color="tab:blue", alpha=0.25, label="± SEM")
plt.scatter(fg_z, ma, s=15, color="tab:blue", label="Mean points", zorder=3)
plt.plot(fg_z, ax_fit, "k--", label="Gaussian fit", zorder=4)
plt.xlim(*Z_RANGE)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "Axial_mean_5beads.svg"), format="pdf")
plt.show()
# ------------------------------------------------
# 7) Lateral plot
# ------------------------------------------------
plt.figure(figsize=(5, 5))
plt.fill_between(fg_l, ml - se_lat, ml + se_lat, color="tab:orange", alpha=0.25, label="± SEM")
plt.scatter(fg_l, ml, s=15, color="tab:orange", label="Mean points", zorder=3)
plt.plot(fg_l, lat_fit, "k--", label="Gaussian fit", zorder=4)
plt.xlim(*L_RANGE)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "Lateral_mean_5beads.svg"), format="pdf")
plt.show()
