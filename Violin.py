import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy.stats import ttest_ind, mannwhitneyu, shapiro

# Constants
Z_SPACING = 5  # micrometers
XY_PIXEL_SIZE_UM = 500.0 / 512.0
OUTPUT_DIR = "/Users/elinestas/Desktop/Box_Plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define session-to-age mapping
session_to_age = {
    'Mouse 1': {'Session 1': 6.5, 'Session 2': 7.5},
    'Mouse 2': {'Session 1': 6.5, 'Session 2': 7.5},
    'Mouse 3': {'Session 1': 8.9, 'Session 2': 9.9}
}

# File paths for vessels
file_paths_v = {
    'Mouse 1': {
        'II': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-58283.1.V-V_fwd.mat',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-58283.2.V-V_fwd.mat',
        }
    },
    'Mouse 2': {
        'II': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-223783.1.V-V_fwd.mat',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-223783.2.V-V_fwd.mat',
        }
    },
    'Mouse 3': {
        'II': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-223785.1.V-V_fwd.mat',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-223785.2.V-V_fwd.mat',
        }
    }
}

# File paths for plaques
file_paths_p = {
    'Mouse 1': {
        'II': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/58283.2.V.npy',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/58283.4.V.npy',
        }
    },
    'Mouse 2': {
        'II': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223783.12.V.npy',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223783.22.V.npy',
        }
    },
    'Mouse 3': {
        'II': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223785.1.V.npy',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223785.2.V.npy'
        }
    }
}

# Define colors for mice
color_palette = {
    'Mouse 1': 'blue',
    'Mouse 2': 'green',
    'Mouse 3': 'orange'
}

# Parameters to analyze
parameters = [
    "Vessel Diameter",
    "Vessel Length",
    "Vessel Tortuosity",
    "Inter-Vessel Distance",
    "Plaque Radius"
]

# -------------------------------------------------------------------
# 1) Normality Check: Shapiro–Wilk
# -------------------------------------------------------------------
def is_normal_distribution(data, alpha=0.05):
    """
    Returns True if data pass the Shapiro–Wilk normality test
    at the specified alpha level, False otherwise.
    For data length < 3, automatically returns True (cannot meaningfully test).
    """
    if len(data) < 3:
        return True
    stat, p_value = shapiro(data)
    return p_value > alpha

# -------------------------------------------------------------------
# 2) Data Loading
# -------------------------------------------------------------------
def load_data_p(file_path):
    """Load plaque data from .npy files."""
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True).item()
    return None

def load_data_v(file_path):
    """Load vessel data from .mat files."""
    if os.path.exists(file_path):
        return sio.loadmat(file_path)
    return None

# -------------------------------------------------------------------
# 3) Parameter Extraction
# -------------------------------------------------------------------
def extract_parameters_v(data):
    return {
        'Vessel Diameter': np.abs(data.get('Diameters', np.array([])).flatten()*XY_PIXEL_SIZE_UM),
        'Vessel Length': np.abs(data.get('Lengths', np.array([])).flatten()*XY_PIXEL_SIZE_UM),
        'Vessel Tortuosity': np.abs(data.get('Tortuosities', np.array([])).flatten()),
        'Inter-Vessel Distance': np.abs(data.get('InterVesselDistances', np.array([])).flatten()*XY_PIXEL_SIZE_UM),
    }

def extract_parameters_p(plaque_data):
    """Extract plaque volume -> radius."""
    if 'plaque_parameters' not in plaque_data:
        return {'Plaque Radius': np.array([])}
    volume = np.abs(plaque_data['plaque_parameters']['Volume (um^3)']).flatten()
    radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
    return {'Plaque Radius': radius}

# -------------------------------------------------------------------
# 4) Load and Process
# -------------------------------------------------------------------
def load_and_process_metrics(area='II'):
    """
    Load plaque and vessel data for each mouse/session, 
    extracting the relevant parameters.
    """
    metrics_by_mouse = {mouse: {} for mouse in file_paths_p}
    for mouse, areas in file_paths_p.items():
        if area not in areas:
            continue

        for session, plaque_file in areas[area].items():
            vessel_file = file_paths_v[mouse][area].get(session, None)
            if not vessel_file or not plaque_file:
                continue

            plaque_data = load_data_p(plaque_file)
            if not plaque_data:
                continue
            plaque_metrics = extract_parameters_p(plaque_data)

            vessel_data = load_data_v(vessel_file)
            if not vessel_data:
                continue
            vessel_metrics = extract_parameters_v(vessel_data)

            # Combine metrics
            metrics_by_mouse[mouse][session] = {
                **plaque_metrics,
                **vessel_metrics
            }
    return metrics_by_mouse

# -------------------------------------------------------------------
# 5) Structure Data for Boxplots
# -------------------------------------------------------------------
def extract_data_for_boxplots(metrics_by_mouse):
    """
    Create a nested dictionary:
    data[param][mouse][session] -> list of values
    """
    data = {param: {mouse: {"Session 1": [], "Session 2": []} for mouse in color_palette}
            for param in parameters}

    for mouse, sessions in metrics_by_mouse.items():
        for session, metrics in sessions.items():
            for param in parameters:
                if param in metrics:
                    data[param][mouse][session].extend(metrics[param])

    return data

# -------------------------------------------------------------------
# 6) Decide: T-test or Mann–Whitney U + Return which test
# -------------------------------------------------------------------
def compare_groups(s1, s2):
    """
    Given two arrays, s1 and s2,
    1) Check normality (Shapiro–Wilk) for each group.
    2) If both normal => use Welch's t-test (ttest_ind with equal_var=False).
       Return (p_val, "Welch t-test").
    3) Otherwise => use Mann–Whitney U test => Return (p_val, "Mann-Whitney U").
    4) If <2 points in either group => return (None, None).
    """
    if len(s1) < 2 or len(s2) < 2:
        return None, None  # Not enough data

    # Check normality
    normal_s1 = is_normal_distribution(s1)
    normal_s2 = is_normal_distribution(s2)

    if normal_s1 and normal_s2:
        # Welch's t-test
        stat, p_val = ttest_ind(s1, s2, equal_var=False)
        return p_val, "Welch t-test"
    else:
        # Mann–Whitney U
        stat, p_val = mannwhitneyu(s1, s2, alternative='two-sided')
        return p_val, "Mann-Whitney U"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# -----------------------------------------------------------------------------
# helper – lighten a colour (used for Session 2)
# -----------------------------------------------------------------------------
def _lighter(c, factor: float = .55):
    r, g, b = mcolors.to_rgb(c)
    return (1 - (1 - r) * factor,
            1 - (1 - g) * factor,
            1 - (1 - b) * factor)

# ------------------------------------------------------------------
# 7) Five-panel publication-style violin plots (seaborn)
#      *zero required arguments beyond the data itself*
# ------------------------------------------------------------------
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_all_violins(data,
                     color_palette=None,
                     compare_groups=None,
                     parameters=None,
                     OUTPUT_DIR=None,
                     age_dict=None):
    """
    Call with just:      plot_all_violins(data_for_boxplots)

    The other arguments fall back to *global* variables of the same names.
    Panels:
      (a) Vessel Diameter       (b) Vessel Length      (c) Vessel Tortuosity
      (d) Inter-Vessel Distance (e) Plaque Radius      (f) Legend
    """
    # ---------- grab globals if caller didn’t pass them --------------------
    if color_palette is None:
        color_palette = globals().get("color_palette")
    if compare_groups is None:
        compare_groups = globals().get("compare_groups")
    if parameters is None:
        parameters = globals().get("parameters")
    if OUTPUT_DIR is None:
        OUTPUT_DIR = globals().get("OUTPUT_DIR", ".")

    if color_palette is None or compare_groups is None or parameters is None:
        raise ValueError("color_palette, compare_groups and parameters must "
                         "exist as globals or be passed explicitly.")

    # ---------- helpers ----------------------------------------------------
    def lighter(c, f=0.55):
        r, g, b = mcolors.to_rgb(c)
        return tuple(1 - (1 - x) * f for x in (r, g, b))

    def p_to_stars(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

    # ---------- config -----------------------------------------------------
    mice          = list(color_palette.keys())
    base_colours  = list(color_palette.values())
    session_names = ["Session 1", "Session 2"]
    params        = parameters[:5]               # first 5 parameters only
    log_y         = {"Vessel Length", "Vessel Tortuosity"}

    order   = [f"{m}_{s}" for m in mice for s in session_names]
    palette = {f"{m}_Session 1": col                for m, col in zip(mice, base_colours)}
    palette.update({f"{m}_Session 2": lighter(col)  for m, col in zip(mice, base_colours)})

    # ---------- canvas -----------------------------------------------------
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    gs  = GridSpec(2, 3, figure=fig, hspace=.55, wspace=.45)
    axes  = [fig.add_subplot(gs[i]) for i in range(5)]
    lg_ax = fig.add_subplot(gs[1, 2]);  lg_ax.axis("off")

    for ax, param, letter in zip(axes, params, ["(a)", "(b)", "(c)", "(d)", "(e)"]):
        # ---- tidy DF ------------------------------------------------------
        rows = []
        for m in mice:
            for s in session_names:
                rows += [{"Value": v, "Mouse": m, "Session": s, "Cat": f"{m}_{s}"}
                         for v in data[param][m][s]]
        df = pd.DataFrame(rows)

        # ---- violin -------------------------------------------------------
        sns.violinplot(ax=ax, data=df, x="Cat", y="Value",
                       order=order, palette=palette,
                       bw=.15, cut=0, width=.30, linewidth=.8,
                       saturation=1, inner="quartile")

        # ---- whisker & bold median ---------------------------------------
        for i, cat in enumerate(order):
            vals = df.loc[df.Cat == cat, "Value"].to_numpy()
            if vals.size == 0:
                continue
            vmin, vmax = vals.min(), vals.max()
            med        = np.median(vals)
            ax.plot([i, i], [vmin, vmax], color="black", lw=.8, zorder=3)
            ax.plot([i-.15, i+.15], [med, med], color="black", lw=2.2, zorder=4)

        # ---- log axis -----------------------------------------------------
        if param in log_y:
            pos = df.Value[df.Value > 0]
            if not pos.empty:
                ax.set_yscale("log")
                ax.set_ylim(bottom=pos.min()*0.5)

        # ---- x-tick relabel ----------------------------------------------
        centres = [0.5 + 2*i for i in range(len(mice))]
        ax.set_xticks(centres)
        ax.set_xticklabels(mice, fontsize=10, weight="bold")

        # ---- cosmetics ----------------------------------------------------
        ax.set_ylabel(f"{param} (µm)", fontsize=11, weight="bold")
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
                fontsize=14, weight="bold", va="center")

        # ---- significance stars ------------------------------------------
        top = ax.get_ylim()[1]
        for m_idx, m in enumerate(mice):
            s1 = df[(df.Mouse == m) & (df.Session == "Session 1")]["Value"]
            s2 = df[(df.Mouse == m) & (df.Session == "Session 2")]["Value"]
            p, _ = compare_groups(s1, s2)
            stars = p_to_stars(p) if p is not None else ""
            if stars:
                x1, x2 = 2*m_idx, 2*m_idx+1
                y = top*0.95 if param not in log_y else top/1.1
                ax.plot([x1, x2], [y, y], color="black", lw=1.2)
                ax.text((x1+x2)/2, y*1.02, stars,
                        ha="center", va="bottom",
                        fontsize=12, weight="bold")

    # ---------- legend -----------------------------------------------------
    legend_handles = []
    for m, col in zip(mice, base_colours):
        age1 = age_dict.get(m, {}).get("Session 1", "S1") if age_dict else "S1"
        age2 = age_dict.get(m, {}).get("Session 2", "S2") if age_dict else "S2"
        legend_handles.append(Patch(color=col,          label=f"{m}: {age1}"))
        legend_handles.append(Patch(color=lighter(col), label=f"{m}: {age2}"))

    lg_ax.legend(handles=legend_handles,
                 title="Age Mice (Months)",
                 frameon=False, fontsize=10, title_fontsize=11,
                 loc="upper left")

    # ---------- export -----------------------------------------------------
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "All_Violins.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Violin panels saved → {out}")

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
def main():
    # 1) Load the data
    metrics_area_II = load_and_process_metrics('II')

    # 2) Extract data for boxplots
    data_for_boxplots = extract_data_for_boxplots(metrics_area_II)

    # 3) Plot the boxplots (no p-value annotations) + print p-values + test name
    plot_all_violins(data_for_boxplots)

if __name__ == "__main__":
    main()
